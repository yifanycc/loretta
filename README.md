<p align="center">
  <img src="logo.png" alt="LoRETTA">
</p>

# Source Code for paper 'LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models'
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


## Quickstart

## Model support
The official supported models are summarized in the table below:
| Models    | layer shape |Loretta shape| 
|-----------| ----------- |----------- |
| Deberta-base    | Title |----------- |
| Roberta-base | Text |----------- |
### Install the required environment
- A conda environment file is stored in the root folder named **'environment.yml'. Create a conda environment with 
this file**
- The project is written mainly with Python 3.10, torch 2.1.2 and the hugging-face transformers 4.28.1
- Set the CUDA_VISIBLE_DEVICES in CUDA_VISIBLE_DEVICES at 'run_all_large_exp.sh'. (Most tests can be fitted within 40G memory, except the Adam full model fine-tuning, which need about 80G memory)
- Change the path in all files (server dir)

## Experiments available and tested in this code

### bert_models
**(run all experiments in 'run_all_bert_exp.sh')**: We offer examples under adam case in the run_all_bert_exp.sh for different PEFT methods base on deberta-base model

Key Notes (changing the arguments for shell functions run_ft_task()):
- models: --model_name_or_path (Have tested: roberta/deberta/albert/bert)
  - the name is the same as the standard name in the huggingface doc
- tuning_type: --tensor_layers and --tuning_type
  - loretta_adp: --tensor_layers=adapters,cls --tuning_type=adapters
  - loretta_rep: --tensor_layers=cls --tuning_type=lore-tt
  - lora: --tensor_layers=None --tuning_type=lora
  - adapters: --tensor_layers=None --tuning_type=adapters
  - ft: --tensor_layers=None --tuning_type=ft
- trainer: --trainer (Have tested: adam/zo)
  - ZO-SGD: --trainer=zo
  - Adam: --trainer=adam
  - ZO-Adam(integrated with torch.optimizer): change the OurTrainer class in run_glue_v5.py with the one in trainer_torch.py
- some parameters are useless, like emb_trunc, linear_trunc, which will be deleted in the future version

### llama_models
**(run all experiments in 'run_all_albert_fine_tune.sh')**:
- Fine-tuning Llama-2-7B model for SST-2 task with Adam full model fine-tuning
- Fine-tuning Llama-2-7B model for SST-2 task with ZO full model fine-tuning
- Fine-tuning Llama-2-7B model for SST-2 task with Adam LoRA fine-tuning
- Fine-tuning Llama-2-7B model for SST-2 task with ZO LoRA fine-tuning **(We recomment this one for ZO training)**

The general setup is similar as bert shell file, the experiment use a low-data resource setting, which only pick up 1000 training samples from the dataset

The code is built base on the github repository of paper 'MeZO: Fine-Tuning Language Models with Just Forward Passes' 
and 'Enabling Lightweight Fine-tuning for Pre-trained Language Model Compression based on Matrix Product Operators'


Please cite the following paper if you are interested in our work or would like to use our code:

@misc{yang2024loretta,
      title={LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models}, 
      author={Yifan Yang and Jiajun Zhou and Ngai Wong and Zheng Zhang},
      year={2024},
      eprint={2402.11417},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
