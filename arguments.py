import logging
import os
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import json
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    TrainingArguments,
)

from transformers.file_utils import cached_property, is_torch_tpu_available

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class MyArguments(TrainingArguments):
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    lm_model: Optional[str] = field(
        default='models/bert-base-uncased',
        metadata={"help": "The LM model name or path for training."},
    )
    train_file: Optional[str] = field(
        default='dataset/ml-1m/train.csv',
        metadata={"help": "The input training data file (a text file)."},
    )
    valid_file: Optional[str] = field(
        default='dataset/ml-1m/valid.csv',
        metadata={"help": "The input validation data file (a text file)."},
    )
    test_file: Optional[str] = field(
        default='dataset/ml-1m/test.csv',
        metadata={"help": "The input test data file (a text file)."},
    )
    cache_dir: Optional[str] = field(
        default='cache',
        metadata={"help": "The cache directory where the model checkpoints will be written."},
    )
    output_dir: Optional[str] = field(
        default='output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    logging_dir: Optional[str] = field(
        default='logs',
        metadata={"help": "Log directory for Tensorboard log output."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X updates steps."},
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "The number of epochs for training."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay if we apply some."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    shuffle_fields: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the fields for lossless augmentation"}
    )
    patience: int = field(
        default=3,
        metadata={"help": "The patience for early stoppint strategy"}
    )
    extend_vocab: str = field(
        # default='raw',
        default=None,
        metadata={"help": "The method to extend the vocabulary. Default to `None` indicating no extension."}
    )

    # CTR's arguments
    neighbor_size: int = field(
        default=64,
        metadata={"help": "the number of triplets in triplet set."}
    )
    n_layer: int = field(
        default=3,
        metadata={"help": "the depth of layer."}
    )
    dim: int = field(
        default=64,
        metadata={"help": "dimension of entity and relation embeddings"}
    )
    n_factor: int = field(
        default=2,
        metadata={"help": "the number of factors"}
    )
    agg: str = field(
        default='decay_sum',
        metadata={"help": "the type of aggregator (sum, pool, concat)"}
    )
    
    # key arguments
    model_mode: str = field(
        default='ctr4lm',
        metadata={"help": "`lm` means LM model only; `ctr` means CTR model only; `ctr4lm` means CTR concat LM."}
    )
    hidden_dim: int = field(
        default=768,
        metadata={"help": "ctr emb_dim ——> hidden dim ——> LM inputs"}
    )
    prompt_model: Optional[str] = field(
        default='models/gpt2',
        metadata={"help": "The model name or path to transform ctr embedding to language prompt."},
    )
    