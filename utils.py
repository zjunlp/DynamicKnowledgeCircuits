import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Define a function to read the JSONL file
def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data = [json.loads(line) for line in lines]
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    

def reduce_array(arr, n):
    """
    Reduce the elements of the sorted array to n elements, evenly reducing the elements.

    Parameters:
    - arr: The original sorted array
    - n: The number of elements after reduction

    Returns:
    - The new reduced array
    """
    N = len(arr)
    if n >= N:
        return arr
    
    step = N / n
    reduced_arr = [arr[int(i * step)] for i in range(n)]
    return reduced_arr


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model: Optional[str] = field(
        default=None,
        metadata={"help": "The model architecture to be trained or fine-tuned."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    load_data_from_cache: bool = field(
        default=False,
        metadata={"help": "Whether not to load data from cache"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`train_file` should be a csv, a json, a jsonl or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`validation_file` should be a csv, a json, a jsonl or a txt file."


@dataclass
class DataPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for prediction.
    """

    input_file: Optional[str] = field(
        default=None, metadata={"help": "The input data file (a text file)."}
    )
    output_file: Optional[str] = field(
        default=None, metadata={"help": "The output file to write the predictions to."}
    )
    batch_size: Optional[int] = field(
        default=16, metadata={"help": "Batch size (default to 16)."}
    )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of samples to predict."}
    )
    max_new_tokens: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "The maximum number of new tokens to generate. "
                "The actual number of new tokens generated may be less."
            )
        },
    )
    num_return_sequences: Optional[int] = field(
        default=1, metadata={"help": "The number of different sequences to generate."}
    )


@dataclass
class ReplayArguments:
    do_replay: Optional[bool] = field(
        default=False, metadata={"help": "Whether to replay the training."}
    )
    replay_file: Optional[str] = field(
        default=None, metadata={"help": "The replay file to replay the training."}
    )
    replay_ratio: Optional[float] = field(
        default=0, metadata={"help": "The replay ratio."}
    )
