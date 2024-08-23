#export HF_HOME='/data/public/yifanyang'
export WANDB_START_METHOD="thread"

# GLUE datasets supported in this code (change in $TASK)
# MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB
# PEFT methods supported in this code (change in $MODE)
# loretta_adp, loretta_rep, lora, adapters (series), prompt, ia3, ptune
# change the number of cuda device by set the $DEVICE

# Test with deberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ft EPOCH=10 BS=8 LR=1e-6 DEVICE=7 bash finetune.sh
MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_adp EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_rep EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=lora EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=adapters EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=prompt EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ia3 EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ptune EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=bitfit EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh

# Test with roberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)

#MODEL=roberta-large TASK=SST2 MODE=ft EPOCH=10 BS=8 LR=1e-6 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=loretta_adp EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=loretta_rep EPOCH=10 BS=8 LR=1e-4 DEVICE=6 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=lora EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=adapters EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=prompt EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ia3 EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ptune EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=bitfit EPOCH=10 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
