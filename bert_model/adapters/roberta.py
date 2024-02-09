import logging

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import ACT2FN, RobertaSelfOutput

from adapters.common import AdapterConfig
from tensor_layers.layers import wrapped_linear_layers

logging.basicConfig(level=logging.INFO)

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))
class RobertaAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(RobertaAdapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)


    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

class RobertaAdapter_tensor(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(RobertaAdapter_tensor, self).__init__()
        tensor_shape = [4, 8, 8, 8, 8, 4]
        tensor_rank = 5
        config_tensor = config_class(shape=tensor_shape, ranks=tensor_rank, set_scale_factors=False)
        self.down_project_tensor = wrapped_linear_layers(in_features=config.hidden_size, out_features=config.adapter_size, tensorized=True, config=config_tensor)
        # nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        # nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project_tensor =wrapped_linear_layers(in_features=config.adapter_size, out_features=config.hidden_size, tensorized=True, config=config_tensor)
        # nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        # nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project_tensor(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project_tensor(activated)
        return hidden_states + up_projected

class RobertaAdapter_SLT(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(RobertaAdapter_SLT, self).__init__()
        tensor_shape = [12, 8, 8, 8, 8, 12]
        tensor_rank = 10
        config_tensor = config_class(shape=tensor_shape, ranks=tensor_rank, set_scale_factors=False)
        self.tensor_layer = wrapped_linear_layers(in_features=config.hidden_size, out_features=config.hidden_size, tensorized=True, config=config_tensor)
        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act


    def forward(self, hidden_states: torch.Tensor):
        output = self.tensor_layer(hidden_states)
        activated = self.activation(output)
        return hidden_states + activated


class RobertaAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: RobertaSelfOutput,
                 config: AdapterConfig):
        super(RobertaAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = RobertaAdapter_tensor(config)
        if config.tensorized:
            self.adapter = RobertaAdapter_tensor(config)
        else:
            self.adapter = RobertaAdapter(config)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapt_roberta_self_output(config: AdapterConfig):
    return lambda self_output: RobertaAdaptedSelfOutput(self_output, config=config)


def add_roberta_adapters(roberta_model: RobertaModel, config: AdapterConfig) -> RobertaModel:
    for layer in roberta_model.encoder.layer:
        layer.attention.output = adapt_roberta_self_output(config)(layer.attention.output)
        layer.output = adapt_roberta_self_output(config)(layer.output)
    return roberta_model


def unfreeze_roberta_adapters(roberta_model: nn.Module, tensorized: bool) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in roberta_model.named_modules():
        if tensorized:
            if isinstance(sub_module, (RobertaAdapter_tensor, nn.LayerNorm)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        else:
            if isinstance(sub_module, (RobertaAdapter, nn.LayerNorm)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

    return roberta_model
