import logging

import torch
import torch.nn as nn
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import ACT2FN, LlamaDecoderLayer,LlamaRMSNorm

from adapters.common import AdapterConfig
from tensor_layers.layers import wrapped_linear_layers
logging.basicConfig(level=logging.INFO)

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))
class LlamaAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(LlamaAdapter, self).__init__()
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

class LlamaAdapter_tensor(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(LlamaAdapter_tensor, self).__init__()
        if config.hidden_size == 768:
            # for deberta-base
            tensor_shape = [8, 8, 12, 8, 8]
        elif config.hidden_size == 1536:
            # for deberta-xxl
            tensor_shape = [4, 8, 8, 8, 8, 6]
        else:
            NotImplementedError
        tensor_rank = 5

        config_tensor = config_class(shape=tensor_shape, ranks=tensor_rank, set_scale_factors=False)
        self.down_project_tensor = wrapped_linear_layers(in_features=config.hidden_size,
                                                         out_features=config.adapter_size, tensorized=True,
                                                         config=config_tensor)
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

class LlamaAdapter_SLT(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(LlamaAdapter_SLT, self).__init__()
        tensor_shape = [12, 8, 8, 8, 8, 12]
        tensor_rank = 5
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


class LlamaMLP_adapter(nn.Module):
    # def __init__(self, config):
    #     super().__init__()
    def __init__(self, self_output,
                 config: AdapterConfig):
        super(LlamaMLP_adapter, self).__init__()
        # self.config = config
        self.self_output = self_output
        if config.tensorized:
            self.adapter = LlamaAdapter_tensor(config)
        else:
            self.adapter = LlamaAdapter(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)

        return hidden_states




def adapt_llama_self_output(config: AdapterConfig):
    return lambda self_output: LlamaMLP_adapter(self_output, config=config)


def add_llama_adapters(LlamaModel: LlamaModel, config: AdapterConfig) -> LlamaModel:
    for layers in LlamaModel.model.layers:
        layers.self_attn.o_proj = adapt_llama_self_output(config)(layers.self_attn.o_proj)
        layers.mlp = adapt_llama_self_output(config)(layers.mlp)
    return LlamaModel


def unfreeze_llama_adapters(LlamaModel: nn.Module, tensorized: bool) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in LlamaModel.named_modules():
        if tensorized:
            if isinstance(sub_module, (LlamaAdapter_tensor, LlamaRMSNorm)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        else:
            if isinstance(sub_module, (LlamaAdapter, LlamaRMSNorm)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
    return LlamaModel
