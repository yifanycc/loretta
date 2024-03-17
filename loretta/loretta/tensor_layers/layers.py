import math
import torch
import numpy as np
import torch.nn as nn
from .low_rank_tensors import TensorTrain, TensorTrainMatrix
from .utils import config_class, quantize, TT_forward_quant


class wrapped_linear_layers(nn.Module):
    def __init__(self,in_features,out_features,bias=True,tensorized=False,config=None):
        super(wrapped_linear_layers,self).__init__()
        if tensorized==True:
            self.layer = TensorizedLinear_module(in_features,out_features, config, bias=bias)
        else: 
            self.layer = torch.nn.Linear(in_features,out_features,bias=bias)
    
        self.tensorized = tensorized
    def forward(self,input,config_forward=None):
        if self.tensorized:
            return self.layer(input,config_forward=config_forward)
        else:
            return self.layer(input)


class TensorizedLinear_module(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                config,
                bias=True
    ):
        """
        config has following attributes:
        shape: the shape of the tensor
        ranks: either a number or a list of numbers to specify the ranks 
        set_scale_factors: True or False
        """

        super(TensorizedLinear_module,self).__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(1/(self.in_features+self.out_features))
        config_tensor = config_class(shape=config.shape,ranks=config.ranks,target_sdv=target_stddev)
        # shape taken care of at input time
        self.tensor = TensorTrain(config_tensor)
        self.tensor_shape = config.shape

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        
        if hasattr(config,'set_scale_factors') and config.set_scale_factors==True:
            self.set_scale_factors()
    
    def set_scale_factors(self,scale_w=1.0,scale_input=1.0,scale_intermediate=1.0,scale_dy=1.0,scale_x=1.0,scale_out=1.0):
        self.scales = torch.nn.ParameterList()
        self.scale_factors = torch.nn.ParameterList()

        if not isinstance(scale_w,list):
            scale_w = [scale_w]*self.tensor.order
        for s in scale_w:
            self.scale_factors.append(torch.nn.Parameter(torch.tensor(s)))

        self.scale_input = torch.nn.Parameter(torch.tensor(scale_input))
        self.scale_intermediate = torch.nn.Parameter(torch.tensor(scale_intermediate))
        self.scale_dy = torch.nn.Parameter(torch.tensor(scale_dy))
        self.scale_x = torch.nn.Parameter(torch.tensor(scale_x))
        self.scale_out = torch.nn.Parameter(torch.tensor(scale_out))


        self.scales.append(self.scale_input)
        self.scales.append(self.scale_intermediate)
        self.scales.append(self.scale_dy)
        self.scales.append(self.scale_x)
        self.scales.append(self.scale_out)


        


    def forward(self,input,config_forward=None):
        """
        config_forward:
        prune_mask: True or False. Use prune mask or not 
        threshold: float number. The threshold to clip rank_parameters to 0
        quantized: 0: full precision. 1: quantization-aware training. 2: low-precision training.
        if quantized:
            rep: INT or FLOAT. quantization type
            bit_input/factors/intermediate/out: bits for each part
            rounding: stochastic or nearest. Rounding type
        """
        if config_forward==None:
            factors =self.tensor.get_factors(prune_mask=False)
        else:
            factors = self.tensor.get_factors(prune_mask=config_forward.prune_mask,threshold=config_forward.threshold)
        

        if config_forward==None or config_forward.quantized==0:
            out = self.forward_tt_full_precision(input,factors)  + self.bias
            

        elif config_forward.quantized == 1:
            out = self.forward_tt_quantization_aware(input,factors,config_forward) + self.bias

        elif config_forward.quantized == 2:
            input = quantize.apply(input,self.scale_input,config_forward.bit_input,config_forward.rep,config_forward.rounding)
            Q_factors = []
            for i,U in enumerate(factors):
                Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.bit_factors,config_forward.rep,config_forward.rounding))
            out = TT_forward_quant.apply(config_forward.rounding,config_forward.bit_intermediate,self.scale_intermediate,self.scale_dy,self.scale_dy,input,*Q_factors).clone()
            out = out + self.bias

        return out 
    

    def forward_tt_quantization_aware(self,input,factors,config_forward):
        # input, scale, bit=[1,7,0],rep='INT',rounding='nearest'
        input = quantize.apply(input,self.scale_input,config_forward.bit_input,config_forward.rep,config_forward.rounding)
        Q_factors = []
        for i,U in enumerate(factors):
            Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.bit_factors,config_forward.rep,config_forward.rounding))
        factors = Q_factors

        quant_intermediate = lambda x: quantize.apply(x,self.scale_intermediate,config_forward.bit_intermediate,config_forward.rep,config_forward.rounding)

        quant_x = lambda x: quantize.apply(x,self.scale_x,config_forward.bit_intermediate,config_forward.rep,config_forward.rounding)

        quant_out = lambda x: quantize.apply(x,self.scale_out,config_forward.bit_out,config_forward.rep,config_forward.rounding)


        m = len(factors)//2
        N = len(input.shape)
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in factors[0:m]]
        elif len(input.shape)==3:
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in factors[0:m]]
        input = torch.reshape(input, [1] + mat_shape)
        

      
        out = factors[0]
        
        out = torch.squeeze(out)

        for i in range(1,m):
            U = factors[i]
            out = quant_intermediate(torch.tensordot(out, U, [[-1],[0]]))


        # S = 100
        out = quant_x(torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))]))

        out = [out] + list(factors[m:])



        N = len(out[0].shape)
        output = factors[m]


        for i in range(m+1,2*m):
            U = factors[i]
            output = quant_intermediate(torch.tensordot(output,U,[[-1],[0]]))
        
        output = torch.tensordot(out[0],output,[[-1],[0]])
        # output = quant_out(output)

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)


        return output


    def forward_tt_full_precision(self, input_mat, factors):
        out = factors[0]
        for i in range(1, len(factors)):
            out = torch.tensordot(out, factors[i], [[-1], [0]])
        output = input_mat @ out.reshape(self.in_features, self.out_features)
        return output



