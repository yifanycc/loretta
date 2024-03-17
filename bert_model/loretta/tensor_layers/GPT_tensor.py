import torch
import torch.nn as nn
import numpy as np

from .GPT_tensor_sublayers import EncoderLayer, Transformer_Embedding
from .layers import wrapped_linear_layers
from .Transformer_tensor_sublayers import Transformer_classifier



class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(self.pos_table[:, :x.size(1),:2])
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding = Transformer_Embedding(config)

        self.encoder_blocks = nn.ModuleList()

        for i in range(config.n_layers):
            self.encoder_blocks.append(EncoderLayer(config))
            
        self.layer_norm = nn.LayerNorm((config.d_model,), eps=1e-12)
    
    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.embedding(input,seg=seg,config_forward=config_forward)

        for layer in self.encoder_blocks:
            output, attn = layer(output,mask=mask,config_forward=config_forward)
        
        output = self.layer_norm(output)
        
        return output


    
class GPT_LM(nn.Module):
    def __init__(self, config, config_LM):
        super(GPT_LM, self).__init__()

        self.encoder = Encoder(config)

        # self.classifier_NEXT = nn.Linear(config.d_model,config.vocab_size)
        self.classifier_NEXT = wrapped_linear_layers(config_LM.d_model, config_LM.vocab_size, tensorized=config_LM.tensorized,config=config_LM, bias=False)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_NEXT = self.classifier_NEXT(output,config_forward=config_forward)


        return output_NEXT

class GPT_pretrain(nn.Module):
    def __init__(self, config, config_next,config_MLM):
        super(GPT_pretrain, self).__init__()

        self.encoder = Encoder(config)

        self.classifier_next = Transformer_classifier(config_next)
        self.classifier_MLM = wrapped_linear_layers(config_MLM.d_model, config_MLM.vocab_size, tensorized=config_MLM.tensorized,config=config_MLM, bias=False)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_next = self.classifier_next(output[:,0,:],config_forward=config_forward)

        output_MLM = self.classifier_MLM(output,config_forward=config_forward)


        return output_next, output_MLM