import logging
import sys
import os
import torch
import torch.nn as nn
from transformers import AlbertModel
from transformers.models.albert.modeling_albert import ACT2FN, AlbertLayer

from adapters.common import AdapterConfig
from tensor_layers.layers import wrapped_linear_layers
# from modeling_albert_tensorized import AlbertAdapter_tensor
logging.basicConfig(level=logging.INFO)
class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))
class AlbertAdapter_tensor(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(AlbertAdapter_tensor, self).__init__()
        tensor_shape = [8, 8, 12, 8, 8]
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


def unfreeze_albert_adapters(albert_model: nn.Module, tensorized: bool) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in albert_model.named_modules():
        if tensorized:
            if isinstance(sub_module, (nn.LayerNorm)) or 'adapter' in name:
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        else:
            NotImplementedError
            # if isinstance(sub_module, (AlbertAdapter, nn.LayerNorm)):
            #     for param_name, param in sub_module.named_parameters():
            #         param.requires_grad = True
    return albert_model
