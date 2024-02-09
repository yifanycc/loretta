import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math
# from tensor_layers.Matrix2MPO import MPO
# from tensor_layers.MPOtorch import LinearDecomMPO
def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        # print(f'train model {mode}')
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        # print(f'check1 factor 0 {self.lora_A[0]}')
        # print(f'check2 factor -1 {self.lora_B[0]}')
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LoRALinear_tt(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_C = nn.Parameter(self.weight.new_zeros((r, 48, 1)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            nn.init.zeros_(self.lora_C)

    def train(self, mode: bool = True):
        # print(f'use')
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        print(f'check1 factor 0 {self.lora_A[0]}')
        print(f'check2 factor 1 {self.lora_B[0]}')
        print(f'check3 factor 2 {self.lora_C[0]}')
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        rec = (self.lora_A @ self.lora_B @ self.lora_C).view(768, 768)
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ rec * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRALinear_tensor(nn.Linear):
    """
    tensorized LoRA implementation for a dense matrix
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.r = 20
        self.lora_alpha = lora_alpha
        shape = [48, 256, 48]
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            print(f'active tensor_lora')
            self.factors_lora = nn.ParameterList(self._build_factors_uniform(shape, r))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    def _build_factors_uniform(self, shape, r):
        factors = []
        order = len(shape)
        if type(r) == int:
            ranks = [1] + [r] * (order - 1) + [1]

        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i + 1]
            U = nn.Parameter(torch.randn(r1, n, r2) / math.sqrt(r2) ** (1 / order), requires_grad=True)
            factors.append(U)
        return factors
            # print(f"i {i} factors {self.factors}")
        # rank adaptive
        # for i in range(1, order):
        #     x = torch.nn.Parameter(torch.ones(ranks[i]))
        #     self.rank_parameters.append(x)

    def reset_parameters(self):
        # set the last factor in the factor list to be zero
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'factors_lora'):
            print(f'check reset')
            for param in self.factors_lora[:-2]:
                nn.init.uniform_(param)
            # nn.init.zeros_(self.factors_lora[-1])
            nn.init.constant_(self.factors_lora[-1], 1e-4)
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        # import tensorly as tl
        # tl.set_backend('pytorch')
        # rec = tl.tt_to_tensor(self.factors_lora).reshape(self.in_features, self.out_features)
        rec = self.mpo.mpo2matrix(self.factors_lora).view(768, 768)

        # for factor in self.factors_lora:
        #     print(f'factor shape {factor.shape}')
        # result = torch.tensordot(self.factors_lora[0], self.factors_lora[1], dims=([2], [0])  )# Contract dimensions (1, 2) of tensor1 with (0, 1) of tensor2
        # result = torch.tensordot(result, self.factors_lora[2], dims=([2], [0]))  #view(768, 768)
        print(f'check1 factor -1 {self.factors_lora[-1][0]}')
        # print(f'check x.shape {x.shape} rec.shape {rec.shape}')
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ rec) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LoRA:

    def __init__(self, model, r, alpha, float16=False, lora_tensor=False):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type in ["albert", "roberta", "deberta"]:
            attention_name = "attention"
        else:
            raise NotImplementedError

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight= attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight 
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type in ["albert"]:
                    original_q_weight = attn.query.weight.data
                    original_q_bias = attn.query.bias.data
                    original_v_weight = attn.value.weight.data
                    original_v_bias = attn.value.bias.data
                    if lora_tensor:
                        attn.query = LoRALinear_tt(model.config.hidden_size, model.config.hidden_size, r=r,
                                                lora_alpha=alpha,
                                                bias=True).to(original_q_weight.device)
                        attn.value = LoRALinear_tt(model.config.hidden_size, model.config.hidden_size, r=r,
                                                lora_alpha=alpha,
                                                bias=True).to(original_v_weight.device)
                    else:
                        attn.query = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                                 bias=True).to(original_q_weight.device)
                        attn.value = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                                 bias=True).to(original_v_weight.device)
                    attn.query.weight.data = original_q_weight
                    attn.query.bias.data = original_q_bias
                    attn.value.weight.data = original_v_weight
                    attn.value.bias.data = original_v_bias
                elif model.config.model_type in ["roberta", "deberta"]:
                    original_q_weight = attn.self.query.weight.data
                    original_q_bias = attn.self.query.bias.data
                    original_v_weight = attn.self.value.weight.data
                    original_v_bias = attn.self.value.bias.data
                    if lora_tensor:
                        attn.self.query = LoRALinear_tensor(model.config.hidden_size, model.config.hidden_size, r=r,
                                                   lora_alpha=alpha,
                                                   bias=True).to(original_q_weight.device)
                        attn.self.value = LoRALinear_tensor(model.config.hidden_size, model.config.hidden_size, r=r,
                                                   lora_alpha=alpha,
                                                   bias=True).to(original_v_weight.device)
                    else:
                        attn.self.query = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                lora_alpha=alpha,
                                                bias=True).to(original_q_weight.device)
                        attn.self.value = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                lora_alpha=alpha,
                                                bias=True).to(original_v_weight.device)
                    attn.self.query.weight.data = original_q_weight
                    attn.self.query.bias.data = original_q_bias
                    attn.self.value.weight.data = original_v_weight
                    attn.self.value.bias.data = original_v_bias
                else:
                    raise NotImplementedError
        
        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False