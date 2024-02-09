
from typing import List, Optional, Tuple, Union
from typing import Optional, Tuple
from torch.nn import functional as F
import tensorly as tl
import sys
sys.path.append('/home/yifanyang/tmp/pycharm_project_668/MPOP/compress_tools_v2')
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from tensor_layers.utils import config_class, quantize, TT_forward_quant
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from tensor_layers.layers import wrapped_linear_layers
from tensor_torch import Tensor, LinearDecomTensor, config_class

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        lm_rank = [1, 16, 256, 144, 9, 1]
        lm_shape = [[8,8,8,8], [10,20,16,10]]
        set_scale_factors = False
        self.config_lm = config_class(shape=lm_shape, ranks=lm_rank, set_scale_factors=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if True:
            self.tensorized = True
            self.tensor_input_shape, self.tensor_output_shape = self.config_lm.shape[0], self.config_lm.shape[1]
            self.lm_head_tensor = LinearDecomTensor(self.tensor_input_shape, self.tensor_output_shape, 5,
                                                tensor_learn=True, use_bias=False)
            # self.from_pretrained_tensor()
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if self.tensorized:
                logits = self.lm_head_tensor(hidden_states)
            else:
                logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def from_pretrained_tensor(self):
        tensor = Tensor(self.tensor_input_shape, self.tensor_output_shape, 5)
        tensor_tensor_set, _, _ = tensor.matrix2tensor(self.lm_head.weight.detach().cpu().numpy())
        tensor_pretrain_weight = [i.flatten() for i in tensor_tensor_set]
        self.lm_head_tensor.from_pretrained((4096, 32000), tensor_pretrain_weight, tensor_tensor_set, use_bias=False)
