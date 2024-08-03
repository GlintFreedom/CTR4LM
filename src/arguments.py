import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    TrainingArguments,
)

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
        default='../models/bert-base-uncased',
        metadata={"help": "The LM model name or path for training."},
    )
    cache_dir: Optional[str] = field(
        default='cache',
        metadata={"help": "The cache directory where the model checkpoints will be written."},
    )
    output_dir: Optional[str] = field(
        default='../output',
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
        default='../logs',
        metadata={"help": "Log directory for Tensorboard log output."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X updates steps."},
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    num_train_epochs: int = field(
        default=50,
        metadata={"help": "The number of epochs for training."},
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
    dataset: str = field(
        default='movie',
        metadata={"help": "Which dataset to use (music, book, paper, movie)."}
    )
    n_epoch: int = field(
        default=20,
        metadata={"help": "The number of epochs for training."}
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for training."}
    )
    inverse_r: bool = field(
        default=True,
        metadata={"help": "Whether to inverse relations in knowledge graph."}
    )
    n_layer: int = field(
        default=2,
        metadata={"help": "The depth of neural network layers."}
    )
    n_factor: int = field(
        default=2,
        metadata={"help": "The number of factors in the model."}
    )
    lr: float = field(
        default=0.001,
        metadata={"help": "The initial learning rate for optimizer."}
    )
    l2_weight: float = field(
        default=1e-4,
        metadata={"help": "Weight of the L2 regularization term."}
    )
    dim: int = field(
        default=64,
        metadata={"help": "Dimension of entity and relation embeddings."}
    )
    neighbor_size: int = field(
        default=64,
        metadata={"help": "The number of triplets in triplet set."}
    )
    agg: str = field(
        default='decay_sum',
        metadata={"help": "The type of aggregator (sum, pool, concat)."}
    )
    use_cuda: bool = field(
        default=True,
        metadata={"help": "Whether to use GPU or CPU for training."}
    )
    show_topk: bool = field(
        default=True,
        metadata={"help": "Whether to show top-k results or not."}
    )
    random_flag: bool = field(
        default=False,
        metadata={"help": "Whether to use a random seed for reproducibility."}
    )
    
    # key arguments
    model_mode: str = field(
        default='ctr',
        metadata={"help": "`lm` means LM model only; `ctr` means CTR model only; `ctr4lm` means CTR concat LM."}
    )
    hidden_dim: int = field(
        default=768,
        metadata={"help": "ctr emb_dim ——> hidden dim ——> LM inputs"}
    )
    prompt_model: Optional[str] = field(
        default='../models/gpt2',
        metadata={"help": "The model name or path to transform ctr embedding to language prompt."},
    )
    