# LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models
Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang

2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)

---

This is the implementation for paper [LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models](https://arxiv.org/pdf/2402.11417.pdf). In this paper,
we propose an ultra parameter efficient fine-tuning method based on Tensor-train decomposition (LoRETTA), which reduce the trainable parameters for up to 100x on the Llama-2-7B model compared with other widely used 
PEFT methods. The paper contains two types of LoRETTA methods, called LoRETTA_adp and LoRETTA_rep, respectively. The LoRETTA_adp
method is suggested to be used in most case, providing on-par or better accuracy with much less trainable parameters. The 
LoRETTA_rep, instead, provided an ultra parameter efficient method that may be beneficial for the future harware design,
which shows superior performance on medium size LLMs like Roberta/Deberta models, while reduce the trainable paraemters to
a great extent. Our implementation is based on the latest huggingface [PEFT](https://github.com/huggingface/peft) package, 
which could be easily plugged-in for most widely used models (see Quickstart and Model Support sections).

Quickstart
---
Install loretta from pip:

```angular2html
pip install loretta
```

Here is a quick example about how to use the loretta package to wrap a huggingface style model with the LoRETTA adapters,
by using the provided `LorettaAdpConfig`,  `LorettaRepConfig`, and `get_peft_model` classes and functions. The general usage
follows the similar logic as the PEFT library. For further introduction about this package, refer to the examples on
bert and llama models at the [official loretta repositories](https://github.com/yifanycc/loretta).


For LoRETTA_adp (we recommend LoRETTA_adp for most cases):
```angular2html
from transformers import AutoModelForCausalLM
from loretta import LorettaAdpConfig, LorettaRepConfig, get_peft_model, TaskType
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
peft_config = LorettaAdpConfig(
                bottleneck_size=64,
                non_linearity="relu",
                adapter_dropout=0.0,
                target_modules=None, # default to be None for official supported models
                scaling=1.0,
                bias="none",
                task_type='CAUSAL_LM', # choose from "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"
                tensor_rank=5,
            )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```
For LoRETTA_rep:

```angular2html
from transformers import AutoModelForCausalLM
from loretta import LorettaAdpConfig, LorettaRepConfig, get_peft_model, TaskType
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
peft_config = LorettaRepConfig(
            r=8, # bottleneck
            lora_alpha=our_args.rep_alpha,
            target_modules=our_args.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=our_args.task_type,
            tensor_rank=our_args.tensor_rank
        )
        model = get_peft_model(model, peft_config)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Model support
---
Generally, we support models with hidden_size of `768, 1024, 1280, 1536, 1600, 2048, 2304, 4096, 5120, 8192`, if you
want to customize for your model, check the specific setting in `utils/tensor_util.py`, `mapping.py` for parameters setup.

Officialy, we have tested on **Deberta (base)**, Roberta (base/large), OPT (1.3B/2.7B/6.7B) and **llama-2 (7B/13B/70B)**.


Environment
---
Generally, the package is implemented base on `torch==2.1.2`, `python=3.10.13` and `transformers==4.38.2`. For a detailed
list of environments we use, check `requirements.txt` or `environment.yml` files we provided.


Cite our paper
---
Note: The code is implemented base on an elder version of the [PEFT library](https://github.com/huggingface/peft/tree/main)

To use loretta in your publication, please cite it by using the following BibTeX entry.
```angular2html
@misc{yang2024loretta,
      title={LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models}, 
      author={Yifan Yang and Jiajun Zhou and Ngai Wong and Zheng Zhang},
      year={2024},
      eprint={2402.11417},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

For more questions, feel free to contact me at `yifanycc@gmail.com` or `yifanycc@gmail.com`.
