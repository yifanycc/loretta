export CUDA_VISIBLE_DEVICES=1
export HF_HOME='/data/public/yifanyang'
export WANDB_START_METHOD="thread"
#SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
##
# zo-tuning (LoRA)
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora LR=1e-4 RANK=8 EPS=1e-3 BS=16 TRAINER=zo bash mezo.sh
#MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=ft LR=1e-4 RANK=8 EPS=1e-3 BS=16 TRAINER=zo bash mezo.sh


# Adam-tuning (FT)
#MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora EPOCH=5 LR=1e-4 RANK=8 EPS=1e-3 BS=16 TRAINER=adam bash finetune.sh
#MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=ft EPOCH=5 LR=1e-4 RANK=8 EPS=1e-3 BS=16 TRAINER=adam bash finetune.sh