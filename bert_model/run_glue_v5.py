import datetime
from datasets import load_dataset
import wandb
import torch
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
os.environ["WANDB_LOG_MODEL"] = "false" # log all model checkpoints
import numpy as np
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, GlueDataset, AutoModelForSequenceClassification, PretrainedConfig, AlbertForSequenceClassification
from transformers import Trainer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

logger = logging.getLogger(__name__)
initial_memory_allocated = torch.cuda.memory_allocated()
initial_memory_cached = torch.cuda.memory_cached()


# Initialize the argument dataclass
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    evaluate_during_training: bool = True
    logging_steps: int = 1000
    lr_scheduler_type: str = 'constant'
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = (
        False  # do not load model by auto device; should turn this on when using FSDP
    )
    wandb_project: str = "camera-ready"
    logging_dir: str = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # parameter setup for PEFT methods
    tuning_type: str = 'ft'
    # LoRETTA
    tensor_rank: int = 5
    target_modules: List[str] = None  # set to be None when use official support model
    task_type: str = 'SEQ_CLS'  # choose from "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"
    # LoRETTA_adp
    adp_bottleneck: int = 64
    non_linearity: str = "relu"
    adapter_dropout: float = 0.0
    scaling: Union[float, str] = 1.0
    # LoRETTA_rep
    rep_bottleneck: int = 8
    rep_alpha: int = 16
    # LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    adapter_size: int = 64
    tensor_shape_opt: int = 0
    report_to: str = "wandb"


def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f'name {name} shape {param.shape} dtype {param.dtype}')
    total_num = sum(p.numel() for p in net.parameters()) / 1000 / 1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000 / 1000
    wandb.log({"Total(M)": total_num, "Trainable(M)": trainable_num})
    return {'Total(M)': total_num, 'Total Trainable(M)': trainable_num}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurArguments))
    model_args, data_args, our_args = parser.parse_args_into_dataclasses()
    if (
            os.path.exists(our_args.output_dir)
            and os.listdir(our_args.output_dir)
            and our_args.do_train
            and not our_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({our_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    wandb_run_name = str(data_args.task_name) + '-' + str(model_args.model_name_or_path.replace('/', '-')) + '-' \
                     + str(our_args.learning_rate)  + '-' \
                     + str(our_args.tuning_type) + '-lorar-' + str(our_args.tensor_rank) + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project=f"<{our_args.wandb_project}>", name=wandb_run_name)
    set_seed(our_args.seed)
    task_name_map = {
        'sst2': 'sst-2',
        'stsb': 'sts-b',
        'mnli': 'mnli',
        'cola': 'cola',
        'qqp': 'qqp',
        'qnli': 'qnli',
        'pte': 'rte',
        'mrpc': 'mrpc',
    }
    data_args.task_name = task_name_map[data_args.task_name]
    # load pretrained model and wrapping model with PEFT methods
    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    if our_args.tuning_type == 'loretta_rep':
        from loretta import LorettaRepConfig, get_peft_model, TaskType
        peft_config = LorettaRepConfig(
            r=our_args.rep_bottleneck,
            lora_alpha=our_args.rep_alpha,
            target_modules=our_args.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=our_args.task_type,
            tensor_rank=our_args.tensor_rank
        )
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'loretta_adp':
        from loretta import get_peft_model, LorettaAdpConfig, TaskType
        peft_config = LorettaAdpConfig(
            bottleneck_size=our_args.adp_bottleneck,
            non_linearity=our_args.non_linearity,
            adapter_dropout=our_args.adapter_dropout,
            target_modules=our_args.target_modules,
            scaling=our_args.scaling,
            bias="none",
            task_type=our_args.task_type,
            tensor_rank=our_args.tensor_rank,
        )
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'lora':
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank,
                                 lora_alpha=our_args.lora_alpha,
                                 lora_dropout=0)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'adapters':
        from peft_local import BottleneckConfig, get_peft_model, TaskType  # noqa: E402
        bottleneck_size: int = 64
        non_linearity: str = "relu"
        adapter_dropout: float = 0.0
        use_parallel_adapter: bool = False
        use_adapterp: bool = False
        target_modules: List[str] = None
        scaling: Union[float, str] = 1.0
        peft_config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if 'Norm' in name:
                param.requires_grad = True
    if our_args.tuning_type == 'bitfit':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    if our_args.tuning_type == 'prompt':
        from peft import get_peft_model, TaskType, PromptTuningConfig
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'ia3':
        from peft import get_peft_model, IA3Config, TaskType
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS, target_modules=None,
            # feedforward_modules=["out_proj"]
        )
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'ptune':
        from peft import get_peft_model, TaskType, PromptEncoderConfig
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=100, encoder_hidden_size=128)
        model = get_peft_model(model, peft_config)

    logger.info("Total Parameter Count: {}M".format(model.num_parameters() / 1000 / 1000))
    logger.info("Total and trainable params: {}".format(str(get_parameter_number(model))))


    # process the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    output_mode = glue_output_modes[data_args.task_name]
    dataset = load_dataset("glue", data_args.task_name.replace("-", ""))
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "sts-b"
        if not is_regression:
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = dataset["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = dataset["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst-2": ("sentence", None),
        "sts-b": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

     # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def tokenize_function(examples):
                # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding='max_length', max_length=128, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
        # return tokenizer(examples['sentence1','sentence2'], truncation=True, padding='max_length',
        #                  max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched" if data_args.task_name == "mnli" else "validation"]
    test_dataset = dataset["test_matched" if data_args.task_name == "mnli" else "test"]
    if data_args.task_name == "qqp":
        subset_size = 1000  # Change this to the desired size of your subset
        eval_dataset = eval_dataset.shuffle(seed=our_args.seed).select([i for i in range(subset_size)])

    # Initialized trainer and evaluation function
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    trainer = Trainer(
        model=model,
        args=our_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    model.eval()
    memory_used_after_part = torch.cuda.memory_allocated() - initial_memory_allocated
    print(f"Memory used after the specific part: {memory_used_after_part / (1024 ** 2)} MB")
    if our_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(our_args.output_dir)

    # Evaluation
    eval_results = {}
    if our_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(
        #         GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        #     )

        for eval_dataset in eval_datasets:
            # trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                our_args.output_dir, f"eval_results_{data_args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(data_args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if our_args.do_predict:

        test_results = {}
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)
            output_test_file = os.path.join(
                our_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


if __name__ == "__main__":
    main()
