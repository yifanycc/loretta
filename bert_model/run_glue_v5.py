# c) Yifan Yang & Jiajun Zhou 2023

import datetime
from datasets import load_dataset
import wandb
import torch
import logging
import os
import sys
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
os.environ["WANDB_LOG_MODEL"] = "true" # log all model checkpoints
import numpy as np
from transformers.models.deberta.modeling_deberta import DebertaLayerNorm
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, GlueDataset, AutoModelForSequenceClassification, PretrainedConfig, AlbertForSequenceClassification
sys.path.append(os.path.join(os.getcwd(), "peft_local/src/"))
from transformers import Trainer
# For ZO training
# from trainer_zo import OurTrainer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from tensor_layers.layers import wrapped_linear_layers

logger = logging.getLogger(__name__)
initial_memory_allocated = torch.cuda.memory_allocated()
initial_memory_cached = torch.cuda.memory_cached()


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
    run_name: str = 'T-adapters'
    trainer: str = field(default='zo', metadata={"help": "which gpu to use"})
    # dataset and sampling strategy
    # task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
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
    wandb_project: str = "bert-base-all"
    logging_dir: str = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = (
        False  # take the log likelihood of all options and train as classification
    )

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = (
        True  # initialize prefix by real activations of random words
    )
    load_best_model_at_end: bool = True
    # fine-tuning type (ft/lora/adapters)
    tuning_type: str = 'ft'
    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA
    lora_tensor: bool = False

    adapter_size: int = 64
    tensor_shape_opt: int = 0
    tensor_rank: int = 5
    report_to: str = "wandb"
    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = (
        False  # use non-differentiable objective (only support F1 for SQuAD for now)
    )

    # Auto saving when interrupted
    save_on_interrupt: bool = (
        False  # save model when interrupted (useful for long training)
    )
    # tensor related
    device_no: str = field(default='3', metadata={"help": "which gpu to use"})
    tensor_layers: str = field(default='word_embed,attention', metadata={"help": "layers need to use tensor format"})


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

    # setup cuda device and logging parameter settings
    os.environ["CUDA_VISIBLE_DEVICES"] = our_args.device_no
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if our_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        our_args.local_rank,
        our_args.device,
        our_args.n_gpu,
        bool(our_args.local_rank != -1),
        our_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", our_args)
    wandb_run_name = str(data_args.task_name) + '-' + str(model_args.model_name_or_path.replace('/', '-')) + '-' \
                     + str(our_args.learning_rate) + '-' + str(our_args.tensor_layers.replace(',', '-')) + '-' \
                     + str(our_args.tuning_type) + '-lorar-' + str(our_args.tensor_rank) + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project=f"<{our_args.wandb_project}>", name=wandb_run_name)
    set_seed(our_args.seed)


    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    # stacking all config together
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    custom_config = our_args.__dict__
    for k, v in custom_config.items():
        setattr(config, k, v)
    config.batch_size = our_args.per_device_train_batch_size
    config.max_seq_length = data_args.max_seq_length
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(model)
    # wrapped the model with adapters
    if our_args.tuning_type == 'lora-tt-cls':
        if model.config.model_type in ["albert"]:
            from lora import LoRA
            LoRA(model, r=our_args.tensor_rank, alpha=our_args.lora_alpha)
        else:
            from peft_local import (  # noqa: E402
                LoraConfig,
                BottleneckConfig,
                PrefixTuningConfig,
                get_peft_model,
                get_peft_model_state_dict,
                prepare_model_for_int8_training,
                set_peft_model_state_dict,
                TaskType,
            )
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank, lora_alpha=our_args.lora_alpha,
                                     lora_dropout=0)
            model = get_peft_model(model, peft_config)
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                if 'pooler' in name:
                    param.requires_grad = True
            # for name, sub_module in model.named_modules():
            #     if isinstance(sub_module, (DebertaLayerNorm)):
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True
            # for name, sub_module in model.named_modules():
            #     if isinstance(sub_module, (nn.LayerNorm)):
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True
    if our_args.tuning_type == 'lora-tt':
        if model.config.model_type in ["albert"]:
            from lora import LoRA
            LoRA(model, r=our_args.tensor_rank, alpha=our_args.lora_alpha)
        else:
            from peft_local import (  # noqa: E402
                LoraConfig,
                BottleneckConfig,
                PrefixTuningConfig,
                get_peft_model,
                get_peft_model_state_dict,
                prepare_model_for_int8_training,
                set_peft_model_state_dict,
                TaskType,
            )
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank, lora_alpha=our_args.lora_alpha,
                                     lora_dropout=0)
            model = get_peft_model(model, peft_config)
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = False
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (DebertaLayerNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (nn.LayerNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
    if our_args.tuning_type == 'lora':
        from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank,
                                 lora_alpha=our_args.lora_alpha,
                                 lora_dropout=0)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'loretta_rep':
        from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, LorettaRepConfig, TaskType
        peft_config = LorettaRepConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank,
                                 lora_alpha=our_args.lora_alpha,
                                 lora_dropout=0, tensorized_cls=True)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'loretta_adp':
        from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, LorettaAdpConfig, TaskType
        peft_config = LorettaAdpConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank,
                                 lora_alpha=our_args.lora_alpha,
                                 lora_dropout=0, tensorized_cls=True)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'lora-cls':
        if model.config.model_type in ["albert", "roberta"]:
            from lora import LoRA
            LoRA(model, r=our_args.tensor_rank, alpha=our_args.lora_alpha)
        else:
            from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=our_args.tensor_rank, lora_alpha=our_args.lora_alpha,
                                     lora_dropout=0)
            model = get_peft_model(model, peft_config)
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (DebertaLayerNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (nn.LayerNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
    if our_args.tuning_type == 'adapters':
        if model.config.model_type == 'albert':
            pass
            from adapters import AdapterConfig, freeze_all_parameters, unfreeze_albert_adapters
            # adapter_config = AdapterConfig(
            #     hidden_size=model.config.hidden_size, adapter_size=our_args.adapter_size,
            #     adapter_act='relu', adapter_initializer_range=1e-2, tensorized=tensorized
            # )
            # model.albert = add_albert_adapters(model.albert, adapter_config)
            model.albert = freeze_all_parameters(model.albert)
            # # Unfreeze adapters and the classifier head
            model.albert = unfreeze_albert_adapters(model.albert, tensorized=tensorized)
            # model.classifier.requires_grad = False
        if model.config.model_type == 'bert':
            from adapters import add_bert_adapters, AdapterConfig, freeze_all_parameters, unfreeze_bert_adapters

            adapter_config = AdapterConfig(
                hidden_size=model.config.hidden_size, adapter_size=our_args.adapter_size,
                adapter_act='relu', adapter_initializer_range=1e-2, tensorized=tensorized, tensor_rank=our_args.tensor_rank
            )
            model.bert = add_bert_adapters(model.bert, adapter_config)
            model.bert = freeze_all_parameters(model.bert)
            # Unfreeze adapters and the classifier head
            model.bert = unfreeze_bert_adapters(model.bert)
            model.classifier.requires_grad = False
        elif model.config.model_type in ['deberta', 'deberta-v2']:
            from adapters import add_deberta_adapters, AdapterConfig, freeze_all_parameters, unfreeze_deberta_adapters
            tensor_shape_list = [[8, 8, 12, 8, 8], [64, 12, 64], [4,4,4,12,4,4,4]]
            tensor_shape = tensor_shape_list[our_args.tensor_shape_opt]
            adapter_config = AdapterConfig(
                hidden_size=model.config.hidden_size, adapter_size=our_args.adapter_size,
                adapter_act='relu', adapter_initializer_range=1e-2, tensorized=tensorized, tensor_rank=our_args.tensor_rank,
                tensor_shape=tensor_shape
            )
            model.deberta = add_deberta_adapters(model.deberta, adapter_config)
            model.deberta = freeze_all_parameters(model.deberta)
            # Unfreeze adapters and the classifier head
            model.deberta = unfreeze_deberta_adapters(model.deberta, tensorized)
            # model.classifier.requires_grad = False

        elif model.config.model_type == 'roberta':
            from adapters import add_roberta_adapters, AdapterConfig, freeze_all_parameters, unfreeze_roberta_adapters
            adapter_config = AdapterConfig(
                hidden_size=768, adapter_size=our_args.adapter_size,
                adapter_act='relu', adapter_initializer_range=1e-2, tensorized=tensorized, tensor_rank=our_args.tensor_rank,
                tensor_shape=[8, 8, 12, 8, 8]
            )
            model.roberta = add_roberta_adapters(model.roberta, adapter_config)
            model.roberta = freeze_all_parameters(model.roberta)
            # Unfreeze adapters and the classifier head
            model.roberta = unfreeze_roberta_adapters(model.roberta, tensorized)
            model.classifier.requires_grad = False
    if our_args.tuning_type == 'bitfit':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    if our_args.tuning_type == 'prompt':
        from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, TaskType, PrefixTuningConfig, \
            PromptTuningConfig
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'ia3':
        from peft_new import get_peft_model, IA3Config, TaskType
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS, target_modules=None,
            # feedforward_modules=["out_proj"]
        )

        model = get_peft_model(model, peft_config)
    if our_args.tuning_type == 'ptune':
        # from prefix import PrefixTuning
        # PrefixTuning(model, num_prefix=5, reparam=False, float16=False, init_by_real_act=True)
        from peft_new import get_peft_config, PeftModel, PeftConfig, get_peft_model, TaskType, PrefixTuningConfig, \
            PromptEncoderConfig
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=100, encoder_hidden_size=128)
        model = get_peft_model(model, peft_config)
        # for name, param in model.named_parameters():
        #     if 'classifier' in name:
        #         param.requires_grad = False
        # for name, param in model.pooler.named_parameters():
        #     param.requires_grad = True
        # for name, param in model.classifier.named_parameters():
        #     param.requires_grad = True

    if 'cls' in config.tensor_layers:
        class config_class():
            def __init__(self,
                         **kwargs):
                for x in kwargs:
                    setattr(self, x, kwargs.get(x))
        if model.config.hidden_size == 1536:
            tensor_shape = [6, 8, 8, 16, 8, 8, 6]
        elif model.config.hidden_size == 768:
            tensor_shape = [12, 8, 8, 8, 8, 12]
        else:
            NotImplementedError
        tensor_rank = our_args.tensor_rank
        config_tensor = config_class(shape=tensor_shape, ranks=tensor_rank, set_scale_factors=False)

        # for name, sub_module in model.named_modules():
        #     if isinstance(sub_module, (DebertaLayerNorm)):
        #         for param_name, param in sub_module.named_parameters():
        #             param.requires_grad = True
        # for name, sub_module in model.named_modules():
        #     if isinstance(sub_module, (nn.LayerNorm)):
        #         for param_name, param in sub_module.named_parameters():
        #             param.requires_grad = True
        if 'deberta' in model.config.model_type:
            model.pooler.dense = wrapped_linear_layers(in_features=model.config.hidden_size, out_features=model.config.hidden_size, tensorized=True, config=config_tensor)
        elif 'roberta' in model.config.model_type:
            model.classifier.dense = wrapped_linear_layers(in_features=model.config.hidden_size,
                                                       out_features=model.config.hidden_size, tensorized=True,
                                                       config=config_tensor)

    # process the dataset
    logger.info("Total Parameter Count: {}M".format(model.num_parameters() / 1000 / 1000))
    logger.info("Total and trainable params: {}".format(str(get_parameter_number(model))))
    task_name_map = {
        'sst-2': 'sst2',
        'sts-b': 'stsb',
        'mnli': 'mnli',
        'cola': 'cola',
        'qqp': 'qqp',
        'qnli': 'qnli',
        'rte': 'rte',
        'mrpc': 'mrpc',
    }

    # if data_args.task_name == 'SST-2':
    print(f'check {data_args.task_name}')
    glue_name = task_name_map[data_args.task_name]
    # Get datasets
    dataset = load_dataset("glue", glue_name)
        # Labels
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
    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

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

    padding = "max_length" if 128 else False



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
    # eval_dataset = dataset["validation"]
    eval_dataset = dataset["validation_matched" if data_args.task_name == "mnli" else "validation"]
    test_dataset = dataset["test_matched" if data_args.task_name == "mnli" else "test"]
    if data_args.task_name == "qqp":
        subset_size = 1000  # Change this to the desired size of your subset
        eval_dataset = eval_dataset.shuffle(seed=our_args.seed).select([i for i in range(subset_size)])


    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn


    # Initialize our Trainer
    if our_args.trainer == 'zo':
        print(f'Using ZO optimizer')
        trainer = Trainer(
            model=model,
            args=our_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name),
            )
    elif our_args.trainer == 'adam':
        print(f'Using adam optimizer')
        our_args.lr_scheduler_type = 'linear'
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


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
