<p align="center">
  <img src="logo.png" alt="LoRETTA">
</p>

# Source Code for paper 'LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models'
2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024 Oral)

Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang

---

This is the implementation for the paper [LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models](https://aclanthology.org/2024.naacl-long.174.pdf). In this paper,
we propose an ultra-parameter efficient fine-tuning method based on Tensor-train decomposition (LoRETTA), which reduces the trainable parameters for up to 100x on the Llama-2-7B model compared with other widely used 
PEFT methods. The paper contains two types of LoRETTA methods, called LoRETTA_adp and LoRETTA_rep, respectively. The LoRETTA_adp
method is suggested to be used in most cases, providing on-par or better accuracy with much less trainable parameters. The 
LoRETTA_rep, instead, provided an ultra parameter efficient method that may be beneficial for future hardware design,
which shows superior performance on medium-size LLMs like Roberta/Deberta models, while reducing the trainable parameters to
a great extent. Our implementation is based on the latest huggingface [PEFT](https://github.com/huggingface/peft) package, 
which could be easily plugged in for the most widely used models (see Quickstart and Model Support sections). 

<h1> <p>🤗 News</p></h1>

**1/09/2025:** There may be a poential bug for bert experiments code, where the previous `glue` repository in Datasets library is no longer available, please replace the dataset name argument to other repository like `nyu-mll/glue` and upgrade both `datasets` and `transformers` library to the latest version for running the code.

**04/21/2024:** Our paper is selected to be an oral paper at NAACL 24 and version 0.1.3 of loretta package has been updated in PyPI, which fix the bugs in the previous version.

**03/17/2024:** The version 0.1.0 package of LoRETTA methods is out, which helps to implement
the LoRETTA_adp and LoRETTA_rep methods with lines of code. Try it by installing with `pip install loretta`

**3/14/2024:** Our paper 'LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models'
has been accepted by the NAACL 2024 main conference

**02/20/2024:** The draft code for reproducing experiments in LoRETTA paper is out. 


Quickstart
---

Install loretta package from pip (require python>3.6, torch>2.0.0, and transformers>4.0.0):

```angular2html
pip install loretta
```

Here is a quick example of how to use the loretta package to wrap a huggingface style model with the LoRETTA adapters,
by using the provided `LorettaAdpConfig`,  `LorettaRepConfig`, and `get_peft_model` classes and functions. The general usage
follows a similar logic as the PEFT library. We mark a symbol `+` for the code you need to add. 


For LoRETTA_adp (we recommend LoRETTA_adp for most cases):
```angular2html
from transformers import AutoModelForCausalLM
+ from loretta import LorettaAdpConfig, LorettaRepConfig, get_peft_model, TaskType
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
+ peft_config = LorettaAdpConfig(
                bottleneck_size=64,
                non_linearity="relu",
                adapter_dropout=0.0,
                target_modules=None, # default to be None for officially supported models
                scaling=1.0,
                bias="none",
                task_type='CAUSAL_LM', # choose from "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"
                tensor_rank=5,
            )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```
For LoRETTA_rep:

```angular2html
from transformers import AutoModelForCausalLM
+ from loretta import LorettaAdpConfig, LorettaRepConfig, get_peft_model, TaskType
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
+ peft_config = LorettaRepConfig(
            r=8, # bottleneck
            lora_alpha=16,
            target_modules=None,
            lora_dropout=0.05,
            bias="none",
            task_type='CAUSAL_LM', # choose from "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"
            tensor_rank=5,
        )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Model support
---
Generally, we support models with hidden_size of `768, 1024, 1280, 1536, 1600, 2048, 2304, 4096, 5120, 8192`, if you
want to customize your model, check the specific setting in `utils/tensor_util.py`, `mapping.py` for parameters setup.

Officially, we have tested on **Deberta (base)**, Roberta (base/large), OPT (1.3B/2.7B/6.7B), and **llama-2 (7B/13B/70B)**.

Note that the LoRETTA library is currently only tested under the FP32 datatype.


Environment
---
Generally, the package is implemented based on `torch==2.1.2`, `python=3.10.13` and `transformers==4.38.2`. For a detailed
list of environments we use, check `requirements.txt` or `environment.yml` files we provided.

Examples of reproducing the results in the paper
---
We provide two detailed examples to reproduce the experimental results in our paper, which are stored in folder `bert_model`
and `large_models`. To reproduce our experiments, follow the instructions below:

- Create the environment with the provided file `requirements.txt` or `environment.yml`
- setup the parameters in `run_all_bert_exp.sh` or `run_all_large_exp.sh` in each folder, which mainly contains:
  - MODEL: the name of the huggingface-enabled model
  - TASK: the name of the datasets, which support `MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB` for bert tests and 
  `SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP` for llama tests
  - EPOCH/BS/LR: basic training argument for epochs, batch_size and learning_rate
  - DEVICE: the number of CUDA devices you would like to use `export CUDA_VISIBLE_DEVICES=$DEVICE`
  - For other arguments needed for the experiments, see `finetune.sh` for detail

Cite our paper
---
Note: The code is implemented based on an elder version of the [PEFT library](https://github.com/huggingface/peft/tree/main)

To use Loretta in your publication, please cite it by using the following BibTeX entry.
```angular2html
@inproceedings{yang2024loretta,
  title={LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models},
  author={Yang, Yifan and Zhou, Jiajun and Wong, Ngai and Zhang, Zheng},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={3161--3176},
  year={2024}
}
```
