set -x
## for jiajun
#base_dir=/home/jjc/project/T-Adapters/tmp/pycharm_project668/albert_tensorized_pub
#data_dir_base=/home/jjc/project/T-Adapters/tmp/pycharm_project668/albert_tensorized_pub
#check_point_dir=/home/jjc/project/T-Adapters/tmp/pycharm_project668/albert_tensorized_pub
#export PYTHONPATH="/home/yifanyang/tmp/T-Adapters:$PYTHONPATH"
export HF_HOME='/data/public/yifanyang'
 # for yifan
base_dir=/home/yifanyang/tmp/T-adapters
data_dir_base=/home/yifanyang/tmp/T-adapters
check_point_dir=/home/yifanyang/tmp/T-adapters
logging_dir_base=/home/yifanyang/tmp/T-adapters/logs
export PYTHONPATH="/home/yifanyang/tmp/T-Adapters:$PYTHONPATH"

gpu_num=$1

echo $gpu_num
#function run_task() {
#  export CUDA_VISIBLE_DEVICES=$1
#  COMMON_ARGS="--data_dir="$data_dir_base/$2" --model_name_or_path=${14} --tokenizer_name=albert-base-v2 --evaluation_strategy=steps --eval_steps=100 --logging_steps=50 --overwrite_output_dir --save_steps=50000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --run_name=$7 --max_seq_length=$8 --tensor_lr=$9 --tensor_layers_tmp=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --max_steps=${15} --load_layer=${16} --update_tensor_layer=${17} ${18}"
#  nohup python run_glue_v5.py \
#      ${COMMON_ARGS} \
#      --do_eval > log_albert/$7.log 2>&1 &
#}
#
function run_ft_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --logging_dir="$logging_dir_base/${14}-$4-${10}-${19}-${20}-$(date +"%Y%m%d%H%M%S")" --model_name_or_path=${14} --tokenizer_name=${14} --evaluation_strategy=steps --eval_steps=500 --logging_steps=50 --overwrite_output_dir --save_steps=10000 --device_no=$1 --task_name=$2 --warmup_step=$3 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --max_seq_length=$8 --tensor_layers=${10} ${18} --trainer=${19} --tuning_type=${20} ${21} ${22} ${23} ${24}"
  python run_glue_v5.py \
      ${COMMON_ARGS} \
      --do_eval
}
function run_cat_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --logging_dir="$logging_dir_base/${14}-$4-${10}-${19}-${20}-$(date +"%Y%m%d%H%M%S")" --model_name_or_path=${14} --tokenizer_name=${14} --evaluation_strategy=steps --eval_steps=500 --logging_steps=50 --overwrite_output_dir --save_steps=10000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --max_seq_length=$8 --tensor_lr=$9 --tensor_layers=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --max_steps=${15} --load_layer=${16} --update_tensor_layer=${17} ${18} --trainer=${19} --tuning_type=${20} ${21} ${22}"
  python run_glue_v5_cat.py \
      ${COMMON_ARGS} \
      --do_eval &
}


function run_pt_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --model_name_or_path=${14} --tokenizer_name=${14} --evaluation_strategy=steps --eval_steps=100 --logging_steps=50 --overwrite_output_dir --save_steps=50000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --run_name=$7 --max_seq_length=$8 --tensor_lr=$9 --tensor_layers=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --max_steps=${15} --load_layer=${16} --update_tensor_layer=${17} ${18} --trainer=${19} --tuning_type=${20}"
  python run_prompt_glue_v5.py \
      ${COMMON_ARGS} \
      --do_eval
}

run_ft_task 1 sst-2 500 1e-5 20 16 sst2 128 2.8e-6 None 480 384 256 kssteven/ibert-roberta-base -1 noload Noupdate --do_train adam ft --max_seq_length=128



#run_ft_task 0,5 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 facebook/opt-1.3b -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 0,5 sst-2 500 5e-4 200.0 128 sst2 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train zo adapters --max_seq_length=128 --lora_r=5 &

#run_ft_task 0,5 sst-2 500 5e-4 200.0 128 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train zo lora --max_seq_length=128 --lora_r=5 &



# deberta-base(T-adapters/tt_cls) - changing tensor shape
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=0
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=1
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=2

#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=1 &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=1 &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=1 &
#
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=2 &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=2 &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=tensor_shape --tensor_shape_opt=2 &
#



# MTL
## T-adapter
#run_cat_task 0 sst-2 500 5e-4 10 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
## T-lora
#run_cat_task 1 sst-2 500 5e-4 10 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
### lora
#run_cat_task 2 sst-2 500 5e-4 10 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
## adapters
#run_cat_task 3 sst-2 500 5e-4 10 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
### ft
##run_cat_task 3 sst-2 500 5e-4 3 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
##


# deberta-base(T-adapters/tt_cls)
#run_ft_task 1 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 2 sst-2 500 5e-4 0.2 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 2 cola 500 5e-4 40.0 32 cola 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 2 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 2 rte 500 5e-4 40.0 16 rte 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 1 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 1 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#
## deberta-base(T-adapters)
# deberta-base(adapters)
#run_ft_task 5 mnli 500 5e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 4 sst-2 500 5e-4 10 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 rte 500 5e-4 40.0 32 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128

# deberta-base(lora)
#run_ft_task 0 mnli 500 5e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 2 rte 500 5e-4 40.0 32 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=5 &

# deberta-base(lora-tt)
#run_ft_task 0 mnli 500 5e-4 20.0 16 mnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 5 cola 500 5e-4 40.0 32 cola 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 5 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 6 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 6 rte 500 5e-4 40.0 16 rte 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 6 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &
#run_ft_task 6 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=30 &


# deberta-base(ft)
#run_ft_task 5 mnli 500 1e-5 20.0 32 mnli 128 2.8e-6 None 480 384 256 roberta-large-mnli -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 3 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 2 rte 500 5e-4 40.0 32 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128

# deberta-xxl(T-adapters/tt_cls)
#run_ft_task 4,5 mnli 500 5e-4 20.0 8 mnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 5 sst-2 500 5e-4 20.0 4 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train zo adapters --max_seq_length=64
#run_ft_task 4,5 cola 500 5e-4 40.0 8 cola 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128

#run_ft_task 4,5 qqp 500 5e-4 20.0 8 qqp 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 4,5 qnli 500 5e-4 20.0 8 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 4,5 rte 500 5e-4 40.0 8 rte 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 4,5 mrpc 500 5e-4 20.0 8 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 4,5 sts-b 500 5e-4 20.0 8 stsb 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128

#
#run_ft_task 0 mnli 500 5e-4 20.0 8 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 sst-2 500 5e-4 20.0 8 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 cola 500 5e-4 40.0 8 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 qqp 500 5e-4 20.0 8 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 qnli 500 5e-4 20.0 8 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 rte 500 5e-4 40.0 8 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 mrpc 500 5e-4 20.0 8 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128
#run_ft_task 0 sts-b 500 5e-4 20.0 8 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128

# deberta-xxl(lora)
#run_ft_task 6 sst-2 500 5e-4 20.0 4 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-v2-xxlarge -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train zo lora --max_seq_length=64 --lora_r=512

## experiment 1118
## deberta-base(T-adapters/tt_cls)
#run_ft_task 1 mnli 0.1 1e-4 20.0 16 mnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 2 sst-2 0.1 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 2 cola 0.1 5e-4 40.0 32 cola 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 2 qqp 0.1 5e-4 20.0 16 qqp 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 1 qnli 0.1 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 2 rte 0.1 1e-4 80.0 16 rte 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 1 mrpc 0.1 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#run_ft_task 1 sts-b 0.1 5e-4 20.0 16 stsb 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &
#
##wait
## deberta-base(T-adapters)
## deberta-base(adapters)
#run_ft_task 4 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 5 cola 500 1e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 5 sts-b 500 1e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
## lora
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 &
#
#wait
#
## deberta-base(ft)
#run_ft_task 4 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 5 sst-2 500 1e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 5 cola 500 1e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 5 qqp 500 1e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 6 qnli 500 1e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 rte 500 1e-4 80.0 32 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 6 mrpc 500 1e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 6 sts-b 500 1e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#
## roberta-base
## deberta-base(T-adapters/tt_cls)
#run_ft_task 4 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 4 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 4 cola 500 5e-4 40.0 32 cola 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 4 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 4 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 5 rte 500 1e-4 80.0 16 rte 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 5 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 5 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#
## deberta-base(T-adapters)
#### deberta-base(adapters)
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 0 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#

#### deberta-base(adapters)
#run_ft_task 1 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
#run_ft_task 1 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ia3 --max_seq_length=128 --adapter_size=8 --wandb_project=rebuttal &
##
## roberta-base(lora/r=4)
#run_ft_task 1 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 1 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
#run_ft_task 2 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=rebuttal &
##
### roberta-base(lora-tt)
#run_ft_task 2 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
#run_ft_task 2 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=rebuttal &
##
#wait
## deberta-base(lora)
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 2 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 2 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 3 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all &
#
##wait
##
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 2 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
##
##wait
##
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 2 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new &
##
#wait
#
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 2 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=bert-base-all-new --adapter_size=8 &
##
### deberta-base(ft)
#run_ft_task 5 mnli 500 1e-4 20.0 32 mnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 5 sst-2 500 1e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 cola 500 1e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 qqp 500 1e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 qnli 500 1e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 rte 500 1e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 mrpc 500 1e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#run_ft_task 4 sts-b 500 1e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ft --max_seq_length=128 &
#
# bitfit
#run_ft_task 2 sst-2 500 1e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 &

# albert-base(T-adapters/tt_cls)
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 0 cola 500 5e-4 40.0 32 cola 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 0 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 6 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 6 rte 500 5e-4 60.0 8 rte 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 6 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#run_ft_task 6 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 adapters,cls 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 &
#
#wait
# albert-base(lora-tt/r=5)
##run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
##run_ft_task 0 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
##run_ft_task 0 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
#run_ft_task 6 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
##run_ft_task 6 rte 500 5e-4 60.0 8 rte 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
#run_ft_task 6 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &
##run_ft_task 6 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 &

#wait
#
### albert-base(lora/r=8)
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 0 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 0 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 6 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 6 rte 500 5e-4 60.0 8 rte 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 6 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &
#run_ft_task 6 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=8 &

#wait
## albert-base(lora/r=4)
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 0 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 0 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 6 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 6 rte 500 5e-4 60.0 8 rte 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 6 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
#run_ft_task 6 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4 &
##wait
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 2 rte 500 5e-4 60.0 8 rte 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --lora_r=4

# exp for overfitting1205
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=albert-base-all &
#run_ft_task 1 qnli 500 5e-4 20.0 16 mnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=albert-base-all &
#
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=albert-base-all &

## exp for multiple ranks
#
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=diff-rank
##run_ft_task 0 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=2 --wandb_project=diff-rank
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=2 --wandb_project=diff-rank
##run_ft_task 3 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=2 --wandb_project=diff-rank
##
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=10 --wandb_project=diff-rank
##run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=10 --wandb_project=diff-rank
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=10 --wandb_project=diff-rank
##run_ft_task 3 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=10 --wandb_project=diff-rank
##
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=20 --wandb_project=diff-rank &
#run_ft_task 3 qnli 500 1e-4 60.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=20 --wandb_project=diff-rank &
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=20 --wandb_project=diff-rank
##run_ft_task 3 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=20 --wandb_project=diff-rank
##
##wait
##
#run_ft_task 1 sst-2 500 5e-4 0.2 16 sst2 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=50 --wandb_project=diff-rank
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --tensor_rank=50 --wandb_project=diff-rank
#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=50 --wandb_project=diff-rank
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --tensor_rank=50 --wandb_project=diff-rank


#run_ft_task 2 cola 0.1 1e-3 40.0 4 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam prompt --max_seq_length=128
#run_ft_task 2 cola 0.1 5e-4 40.0 32 cola 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam ptune --max_seq_length=128


#
## 1207
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=8 --wandb_project=bert-base-all &
#run_ft_task 1 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=8 --wandb_project=bert-base-all &
#
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project=bert-base-all &
#run_ft_task 2 rte 500 5e-4 80.0 16 rte 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora --max_seq_length=128 --tensor_rank=4 --wandb_project==bert-base-all &
#
#
### deberta-base(T-adapters/tt_cls)
#run_ft_task 2 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 cola 500 5e-4 40.0 32 cola 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 0 rte 500 5e-4 80.0 16 rte 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 0 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 0 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 adapters,cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=roberta-base &
##
#wait
#
#run_ft_task 0 mnli 500 1e-4 20.0 16 mnli 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 1 cola 500 5e-4 40.0 32 cola 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 1 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 2 rte 500 5e-4 40.0 16 rte 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 sts-b 500 5e-4 20.0 16 stsb 128 2.8e-6 cls 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --wandb_project=roberta-base &
#
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam bitfit --max_seq_length=128 --wandb_project=roberta-base &
#run_ft_task 3 qqp 500 5e-4 20.0 16 qqp 128 2.8e-6 None 480 384 256 roberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-cls --tensor_rank=8 --max_seq_length=128 --wandb_project=roberta-base &

#run_ft_task 2 rte 500 1e-3 80.0 16 rte 128 2.8e-6 adapters,cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam adapters --max_seq_length=128 --wandb_project=our-1206 &

# test

#run_ft_task 0 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 1 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 cls 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
##
#wait
##
#run_ft_task 2 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 2 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &

#wait

#run_ft_task 1 sst-2 500 5e-4 20.0 16 sst2 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt-cls --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 2 qnli 500 5e-4 20.0 16 qnli 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt-cls --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &
#run_ft_task 3 mrpc 500 5e-4 20.0 16 mrpc 128 2.8e-6 None 480 384 256 microsoft/deberta-base -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train adam lora-tt-cls --max_seq_length=128 --lora_r=5 --wandb_project=diff-modules &

