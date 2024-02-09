import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TensorizedLinear_module, TensorizedEmbedding, TensorizedEmbedding_order4
from .utils import config_class
from .layers import wrapped_linear_layers

"""
config for Transformer:
n_layers: number of encoder blocks in the model
vocab_size: total number of vocab for embedding
n_position: number of positions
d_model: number of model dimension
d_hid: hidden number of parameters in PFF layers
n_head: number of heads
num_class: number of classes
dropout: dropout prob
tensorized: True or False
embedding: contains tensor setting for four linear layers in classifier.
pff: {0,1},contains tensor setting for two linear layers in PFF.
attn: {q,k,v,fc} contains tensor setting for four linear layers in ATTN.
classification: contains tensor setting for four linear layers in classifier.
"""

"""
config for tensor settings has following attributes:
shape: the shape of the tensor
ranks: either a number or a list of numbers to specify the ranks 
set_scale_factors: True or False
"""


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

# def create_linear_layers(in_features,out_features,bias=True,tensorized=False,config=None):
#     if tensorized==True:
#         return TensorizedLinear_module(in_features,out_features, config, bias=bias)
#     else: 
#         return torch.nn.Linear(in_features,out_features,bias=bias)
    

# class wrapped_linear_layers(nn.Module):
#     def __init__(self,in_features,out_features,bias=True,tensorized=False,config=None):
#         super(wrapped_linear_layers,self).__init__()
#         if tensorized==True:
#             self.layer = TensorizedLinear_module(in_features,out_features, config, bias=bias)
#         else: 
#             self.layer = torch.nn.Linear(in_features,out_features,bias=bias)
    
#         self.tensorized = tensorized
#     def forward(self,input,config_forward=None):
#         if self.tensorized:
#             return self.layer(input,config_forward=config_forward)
#         else:
#             return self.layer(input)
        

    
class Transformer_Embedding(nn.Module):
    def __init__(self,config):
        super(Transformer_Embedding,self).__init__()
        if config.tensorized==True:
            if len(config.embedding.shape[0])==4:
                self.word_emb = TensorizedEmbedding_order4(config.vocab_size,config.d_model,config.embedding)
            else:
                self.word_emb = TensorizedEmbedding(config.vocab_size,config.d_model,config.embedding)
        else:
            self.word_emb = torch.nn.Embedding(config.vocab_size,config.d_model)

        self.position_emb = nn.Embedding(config.n_position,config.d_model)

        self.register_buffer("position_ids", torch.arange(config.n_position).expand((1, -1)))

        # self.token_type_emb = None
        # if config.token_type_emb==True:
        self.token_type_emb = nn.Embedding(2,config.d_model)

        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(p=config.dropout)
        self.tensorized = config.tensorized
    def forward(self,input,seg=None,config_forward=None):
        position_ids = self.position_ids[:, 0: input.shape[1]]
        if self.tensorized:
            enc_output = self.word_emb(input,config_forward) + self.position_emb(position_ids)
        else:
            enc_output = self.word_emb(input) + self.position_emb(position_ids)

        if seg!=None: 
            enc_output += self.token_type_emb(seg)
        
        enc_output = self.dropout(self.layer_norm(enc_output))

        return enc_output

        

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.slf_attn = GPT_Attention(config)
        self.pos_ffn = GPT_PFF(config)
        
        
        self.layer_norm_attn = nn.LayerNorm((config.d_model,), eps=1e-12)
        self.layer_norm_pff = nn.LayerNorm((config.d_model,), eps=1e-12)
        

        
        
       
    def forward(self, enc_input, mask=None, config_forward=None):
        residual = enc_input
        enc_input = self.layer_norm_attn(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=mask, config_forward=config_forward)
        enc_output = enc_output + residual
        
        residual = enc_output
        enc_output = self.layer_norm_pff(enc_output)
        enc_output = self.pos_ffn(enc_output,config_forward=config_forward)
        enc_output = enc_output + residual

        return enc_output, enc_slf_attn



class GPT_PFF(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, config):
        super().__init__()
        

        # self.fc_1 = TensorizedLinear_module(config.d_model, config.d_hid, config.pff[0], bias=True)
        # self.fc_2 = TensorizedLinear_module(config.d_hid, config.d_model, config.pff[1], bias=True)

        self.fc_1 = wrapped_linear_layers(config.d_model, config.d_hid, tensorized=config.tensorized,config=config.pff[0], bias=True)
        self.fc_2 = wrapped_linear_layers(config.d_hid, config.d_model, tensorized=config.tensorized,config=config.pff[1], bias=True)
        


        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x,config_forward=None):
      


        x = self.fc_1(x,config_forward=config_forward)
        x = self.act(x)
        x = self.fc_2(x,config_forward=config_forward)

        x = self.dropout(x)
   

        
        return x



class GPT_Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,config):


    # def __init__(self, d_model,d_q,d_k,d_v, n_head, shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], rank=[20,20],tensor_type = 'TensorTrain', dropout=0.1,
    #             bit_w = 8, scale_w = 2**(-5), 
    #             quantized = False):
        super().__init__()

        d_model = config.d_model
        n_head = config.n_head
        d_q, d_k, d_v = d_model//n_head, d_model//n_head, d_model//n_head

        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head



        # self.w_qs = TensorizedLinear_module(d_model, d_q*n_head, config.attn['q'], bias=True)
        # self.w_ks = TensorizedLinear_module(d_model, d_k*n_head, config.attn['k'], bias=True)
        # self.w_vs = TensorizedLinear_module(d_model, d_v*n_head, config.attn['v'], bias=True)

        # self.fc = TensorizedLinear_module(d_model, d_v*n_head, config, bias=True)

        self.w_qs = wrapped_linear_layers(d_model, d_q*n_head, tensorized=config.tensorized,config=config.attn['q'], bias=True)
        self.w_ks = wrapped_linear_layers(d_model, d_k*n_head, tensorized=config.tensorized,config=config.attn['k'], bias=True)
        self.w_vs = wrapped_linear_layers(d_model, d_v*n_head, tensorized=config.tensorized,config=config.attn['v'], bias=True)

        self.fc = wrapped_linear_layers(d_model, d_v*n_head, tensorized=config.tensorized,config=config.attn['fc'], bias=True)


        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)


        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm((d_model,), eps=1e-12)



    

    def forward(self, q, k, v, mask=None, config_forward=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)


 

        q = self.w_qs(q,config_forward=config_forward).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k,config_forward=config_forward).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v,config_forward=config_forward).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        
        




        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)




        q = self.dropout(self.fc(q,config_forward=config_forward))




        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
            if len(mask.shape)==3:
                mask = mask.unsqueeze(1)
            
  
            mask.to(torch.float32)
            mask = (1-mask)*(-1e9)

        
            attn = attn.to(torch.float32) + mask



        attn_prob = F.softmax(attn, dim=-1).to(v.dtype)
        attn_prob = self.dropout(attn_prob)



        output = torch.matmul(attn_prob, v)


        return output, (attn_prob,attn)
