import logging
import math
import os
import sys
import random
import numpy as np
import pandas as pd
from itertools import chain

import wandb
import datasets
from datasets import load_dataset, Dataset, DatasetDict

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from utils import read_jsonl, ModelArguments, DataTrainingArguments, ReplayArguments


os.environ["WANDB_PROJECT"] = "Forgetting"

logger = logging.getLogger(__name__)

MASK_TYPE = {
    "new": 1,
    "revised": 2,
}


class ExponentialSaveCallback(transformers.TrainerCallback):
    def __init__(self, initial_step=1, base=2, max_step=16):
        self.initial_step = initial_step  # 第一个保存步骤
        self.base = base  # 指数基数
        self.max_step = max_step  # 最大保存步数上限
        self.saved_steps = [0]  # 用来记录已保存的步骤

    def on_step_end(self, args, state, control, **kwargs):
        # 获取当前步骤
        step = state.global_step

        # 计算下一个保存步数，指数增长
        target_step = self.initial_step * (
            self.base
            ** (len(self.saved_steps[:-1]) - int(len(self.saved_steps[:-1]) / 2))
        )

        control.should_save = False

        # 如果步骤小于或等于max_step，则进行保存
        if target_step <= self.max_step and step >= self.saved_steps[-1] + target_step:
            # 进行保存并记录已保存的步骤
            self.saved_steps.append(step)
            print(f"Saving model at step {step} (exponentially based save)")

            # 触发保存
            control.should_save = True
        elif target_step > self.max_step:
            # 一旦超过最大步数限制，继续按 max_step 保存
            if step >= self.saved_steps[-1] + self.max_step:
                self.saved_steps.append(step)
                print(
                    f"Saving model at step {step} (max_step reached, constant saving)"
                )
                control.should_save = True  # 持续保存
        return control


class LoggingCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logger.info(
                f"step: {state.global_step}, loss: {logs.get('loss')}, grad_norm: {logs.get('grad_norm')}, learning_rate: {logs.get('learning_rate')}, epoch: {state.epoch}"
            )


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop("first_token_attribute_mask")
        return super().compute_loss(model, inputs, return_outputs)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, ReplayArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, replay_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, replay_args = (
            parser.parse_args_into_dataclasses()
        )

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.logging_dir, "train.log")),
        ],
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if replay_args.do_replay and replay_args.replay_file is not None:
        replay_data = read_jsonl(replay_args.replay_file)
    else:
        replay_data = None

    def replace_with_random_by_ratio(A, B, r):
        """
        将数组 A 中比例为 r 的元素替换为数组 B 中的随机元素。

        参数:
            A (list): 原数组 A。
            B (list): 替换来源数组 B。
            r (float): 替换比例，取值范围为 0 到 1。

        返回:
            list: 替换后的数组 A。
        """
        if not (0 <= r <= 1):
            raise ValueError("r 必须在 0 到 1 之间")

        if r == 0:
            return A

        num_replace = int(len(A) * r)  # 计算需要替换的元素个数
        indices_to_replace = random.sample(
            range(len(A)), num_replace
        )  # 随机选择替换的索引

        for idx in indices_to_replace:
            A[idx] = random.choice(B)  # 从 B 中随机选择元素替换 A 中的元素

        return A

    # Get the datasets: you can either provide your own CSV/JSON/JSONL/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        if extension == "text" or extension == "json" or extension == "csv":
            if "validation" in data_files:
                raw_datasets = load_dataset(
                    extension,
                    data_files=data_files,
                    **dataset_args,
                )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )
        elif extension == "jsonl":
            train_data = read_jsonl(data_files.get("train"))
            if replay_args.do_replay:
                assert replay_data is not None
                train_data = replace_with_random_by_ratio(
                    train_data, replay_data, replay_args.replay_ratio
                )
            train_df = pd.DataFrame(train_data)
            train_df = train_df.fillna("").astype(str)
            train_dataset = Dataset.from_pandas(train_df)
            if "validation" in data_files:
                eval_data = read_jsonl(data_files["validation"])
                eval_df = pd.DataFrame(eval_data)
                eval_df = eval_df.fillna("").astype(str)
                eval_dataset = Dataset.from_pandas(eval_df)
                raw_datasets = DatasetDict(
                    {
                        "train": train_dataset,
                        "validation": eval_dataset,
                    }
                )
            else:
                splited_datasets = train_dataset.train_test_split(
                    test_size=data_args.validation_split_percentage / 100
                )
                raw_datasets = DatasetDict(
                    {
                        "train": splited_datasets["train"],
                        "validation": splited_datasets["test"],
                    }
                )

    # Load pretrained tokenizer and config
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Load pretrained model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # Initialize a new model from scratch
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 2048 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 2048
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def find_subarray(arr, subarr):
        arr = np.array(arr)
        subarr = np.array(subarr)

        for i in range(len(arr) - len(subarr) + 1):
            if np.array_equal(arr[i : i + len(subarr)], subarr):
                return i
        return -1

    def tokenize(prompt, add_eos_token=True):
        cutoff_len = block_size
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_function(data_point):
        output = tokenize(data_point[text_column_name])
        first_token_attribute_mask = [0] * len(output["input_ids"])
        for column in [
            "birthday",
            "city",
            "company",
            "major",
            "university",
        ]:
            if data_point.get(column) is not None and data_point.get(column) != "":
                attribute = f" {data_point[column]}"
                tokenized_attribute = tokenizer(attribute, add_special_tokens=False)
                if (
                    tokenized_attribute["input_ids"][0]
                    == tokenizer(" ", add_special_tokens=False)["input_ids"][0]
                ):
                    for key, value in tokenized_attribute.items():
                        tokenized_attribute[key] = value[1:]
                attribute_idx = find_subarray(
                    output["input_ids"], tokenized_attribute["input_ids"]
                )
                if attribute_idx != -1:
                    first_token_attribute_mask[attribute_idx] = MASK_TYPE[
                        data_point["type"]
                    ]
        output["first_token_attribute_mask"] = first_token_attribute_mask
        return output

    data_dir = os.path.dirname(data_args.train_file)
    cache_tokenized_datasets_path = os.path.join(
        data_dir, "cache", model_args.model, "tokenized_datasets"
    )
    os.makedirs(cache_tokenized_datasets_path, exist_ok=True)
    if data_args.load_data_from_cache:
        tokenized_datasets = datasets.load_from_disk(cache_tokenized_datasets_path)
    else:
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        tokenized_datasets = tokenized_datasets.shuffle(seed=training_args.seed)
        tokenized_datasets.save_to_disk(cache_tokenized_datasets_path)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        start_idx_list = [0]
        start_idx_cnt = 0
        for item in list(examples.values())[0]:
            if start_idx_cnt + len(item) > start_idx_list[-1] + block_size:
                start_idx_list.append(start_idx_cnt)
            start_idx_cnt += len(item)

        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in start_idx_list[:-1]]
            for k, t in concatenated_examples.items()
        }
        return result

    cache_lm_datasets_path = os.path.join(
        data_dir, "cache", model_args.model, "lm_datasets"
    )
    os.makedirs(cache_lm_datasets_path, exist_ok=True)
    if data_args.load_data_from_cache:
        lm_datasets = datasets.load_from_disk(cache_lm_datasets_path)
    else:
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        lm_datasets = lm_datasets.shuffle(seed=training_args.seed)
        lm_datasets.save_to_disk(cache_lm_datasets_path)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    training_args.label_names = [
        "labels",
        "first_token_attribute_mask",
    ]

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        accuracy_metric = evaluate.load("./metrics/accuracy")

        def compute_metrics(eval_preds):
            preds, label_columns = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = label_columns[0][:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            first_token_attribute_masks = label_columns[1][:, 1:].reshape(-1)
            accuracy = accuracy_metric.compute(predictions=preds, references=labels)

            new_first_token_attribute_accuracy = accuracy_metric.compute(
                predictions=preds[first_token_attribute_masks == MASK_TYPE["new"]],
                references=labels[first_token_attribute_masks == MASK_TYPE["new"]],
            )
            revised_first_token_attribute_accuracy = accuracy_metric.compute(
                predictions=preds[first_token_attribute_masks == MASK_TYPE["revised"]],
                references=labels[first_token_attribute_masks == MASK_TYPE["revised"]],
            )

            return {
                "accuracy": accuracy["accuracy"],
                "new_first_token_attribute_accuracy": new_first_token_attribute_accuracy[
                    "accuracy"
                ],
                "revised_first_token_attribute_accuracy": revised_first_token_attribute_accuracy[
                    "accuracy"
                ],
            }

    # Setup lr_scheduler_kwargs in TrainingArguments
    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {"min_lr": float(1e-5)}

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=(compute_metrics if training_args.do_eval else None),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
        callbacks=[ExponentialSaveCallback(), LoggingCallback()],
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
