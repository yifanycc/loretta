#export HF_HOME='/data/public/yifanyang'
export WANDB_START_METHOD="thread"

# datasets supported in this code (change in $TASK)
# SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
# PEFT methods supported in this code (change in $MODE)
# loretta_adp, loretta_rep, lora, adapters (series), prompt, ia3, ptune

# test examples (for SST2 task)
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=loretta_adp EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=loretta_rep EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=adapters EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=prompt EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=ia3 EPOCH=3 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
