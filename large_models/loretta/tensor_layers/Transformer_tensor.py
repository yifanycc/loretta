import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .Transformer_tensor_sublayers import EncoderLayer, Transformer_Embedding, Transformer_classifier
from .layers import TensorizedEmbedding, TensorizedLinear_module, wrapped_linear_layers


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
    
    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.embedding(input,seg=seg,config_forward=config_forward)

        for layer in self.encoder_blocks:
            output, attn = layer(output,mask=mask,config_forward=config_forward)
        
        return output
# the encoder is same as the BertModel() without pooler layer
class Transformer_classification(nn.Module):
    def __init__(self, config, config_classifiction):
        super(Transformer_classification, self).__init__()

        self.encoder = Encoder(config)

        self.classifier = Transformer_classifier(config_classifiction)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output = output[:,0,:]

        output = self.classifier(output,config_forward=config_forward)

        return output


class Transformer_classification_SLU(nn.Module):
    def __init__(self, config, config_intent,config_slot):
        super(Transformer_classification_SLU, self).__init__()

        self.encoder = Encoder(config)

        self.classifier = Transformer_classifier(config_intent)
        self.slot_classifier = Transformer_classifier(config_slot)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_intent = self.classifier(output[:,0,:],config_forward=config_forward)

        output_slot = self.slot_classifier(output[:,1:,:],config_forward=config_forward)


        return output_intent, output_slot

class Transformer_pretrain(nn.Module):
    def __init__(self, config, config_next,config_MLM):
        super(Transformer_pretrain, self).__init__()

        self.encoder = Encoder(config)

        self.classifier_next = Transformer_classifier(config_next)
        self.classifier_MLM = Transformer_classifier(config_MLM)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_next = self.classifier_next(output[:,0,:],config_forward=config_forward)

        output_MLM = self.classifier_MLM(output,config_forward=config_forward)


        return output_next, output_MLM


class Transformer_pretrain_new(nn.Module):
    def __init__(self, config, config_next,config_MLM):
        super(Transformer_pretrain_new, self).__init__()

        self.encoder = Encoder(config)

        self.classifier_next = Transformer_classifier(config_next)
        self.classifier_MLM = wrapped_linear_layers(config_MLM.d_model, config_MLM.vocab_size, tensorized=config_MLM.tensorized,config=config_MLM, bias=False)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_next = self.classifier_next(output[:,0,:],config_forward=config_forward)

        output_MLM = self.classifier_MLM(output,config_forward=config_forward)


        return output_next, output_MLM
    
class Transformer_NextWordPrediction(nn.Module):
    def __init__(self, config, config_NEXT):
        super(Transformer_NextWordPrediction, self).__init__()

        self.encoder = Encoder(config)

        self.classifier_NEXT = Transformer_classifier(config_NEXT)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_NEXT = self.classifier_NEXT(output,config_forward=config_forward)


        return output_NEXT


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)


        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class Transformer_pretrain_BERT(nn.Module):
    def __init__(self, config, config_next,config_MLM):
        super(Transformer_pretrain_BERT, self).__init__()

        self.encoder = Encoder(config)

        self.classifier_next = nn.Linear(config_next.d_model,2)
        self.classifier_MLM = BertLMPredictionHead(config_MLM)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output_next = self.classifier_next(output[:,0,:])

        output_MLM = self.classifier_MLM(output)


        return output_next, output_MLM

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForSequenceClassification_tensor(nn.Module):
    def __init__(self, config, config_forward):
        super(BertForSequenceClassification_tensor, self).__init__()
        self.config_forward = config_forward
        self.encoder = Encoder(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(768, 2)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        output = self.encoder(input_ids, mask=attention_mask, seg=None, config_forward=self.config_forward)
        pooled_output = self.pooler(output)
        logits = self.classifier(pooled_output)
        # print(f'checkpoint: output shape {output.shape} pooled out {pooled_output.shape} logits shape {logits.shape} label shape {labels.shape}')
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    # def forward(self,input,mask=None,seg=None,config_forward=None):
    #     output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)
    #
    #     output_intent = self.classifier(output[:,0,:],config_forward=config_forward)
    #
    #     output_slot = self.slot_classifier(output[:,1:,:],config_forward=config_forward)
    #
    #
    #     return output_intent, output_slot