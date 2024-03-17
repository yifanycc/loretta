
# datasets supported in this code (change in $TASK)
# MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB
# PEFT methods supported in this code (change in $MODE)
# loretta_adp, loretta_rep, lora, adapters (series), prompt, ia3, ptune
export HF_HOME='/data/public/yifanyang'
export WANDB_START_METHOD="thread"

#MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_adp EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=loretta_rep EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=lora EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=adapters EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=prompt EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ia3 EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=ptune EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=microsoft/deberta-base TASK=SST2 MODE=bitfit EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh

#MODEL=roberta-large TASK=SST2 MODE=loretta_adp EPOCH=5 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
MODEL=roberta-large TASK=SST2 MODE=loretta_rep EPOCH=0.01 BS=8 LR=1e-4 DEVICE=6 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=lora EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=adapters EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=prompt EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ia3 EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=ptune EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh
#MODEL=roberta-large TASK=SST2 MODE=bitfit EPOCH=0.01 BS=8 LR=1e-4 DEVICE=7 bash finetune.sh