
# Source Code for paper 'LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models'

## Paper under review by 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)

## We recommend run the Llama-2 tasks, which is faster to converge under the given low data resource setting

## Install the required environment
- An conda environment file is stored in the root folder named **'environment.yml'. Create conda environment with 
this file**
- The project is written mainly with python 3.10, torch 2.1.2 and the hugging-face transformers 4.28.1
- Set the CUDA_VISIBLE_DEVICES in CUDA_VISIBLE_DEVICES at 'run_all_large_exp.sh'. (Most test can be fitted within 40G memory, except the Adam full model fine-tuning, which need about 80G memory)
- Change the path in all files (server dir)

## Experiments available and tested in this code

### bert_models
**(run all experiments in 'run_all_bert_exp.sh')**: We offer examples under adam case in the run_all_bert_exp.sh for different PEFT methods base on deberta-base model

Key Notes (changing the arguments for shell functions run_ft_task()):
- models: --model_name_or_path (Have tested: roberta/deberta/albert/bert)
  - the name is same as the standard name in the huggingface doc
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
