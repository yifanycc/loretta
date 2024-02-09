from typing import NamedTuple, Union, Callable, List

import torch.nn as nn


class AdapterConfig(NamedTuple):
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float
    tensorized: bool
    tensor_rank: int
    tensor_shape: List

def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model
