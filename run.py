import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from arguments import MyArguments
from transformers import (
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    BertModel,
)
import torch.nn as nn
from dataset import load_csv_as_df, MyDataset, MyDataCollator
from mymodel import MyModel

logger = logging.getLogger(__name__)


def showTokenizer(tokenizer):
    # show tokenizer info
    logger.info(
        "\n\tTokenizer type: %s\n\tPL's path: %s\n\tSize of vocab: %d\n\tMax length of input: %d\n\tSpecial tokens: %s" %
        (
            tokenizer.__class__.__name__,
            tokenizer.name_or_path,
            len(tokenizer.get_vocab()),
            tokenizer.model_max_length,
            ", ".join([f"{key}: {value}" for key, value in tokenizer.special_tokens_map.items()])
        )
    )


def vocabulary_extension(tokenizer, df_list, extension_method, extension_fields):
    if extension_method == "raw":
        new_tokens = set()
        for df in df_list:
            for field in extension_fields:
                new_tokens.update(df[field].unique())

        new_tokens = list(new_tokens)
        tokenizer.add_tokens(new_tokens)

    return tokenizer


def main():
    parser = HfArgumentParser(MyArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename='',
        filemode='a'
    )
    file_handler = logging.FileHandler('logs/test.log')
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    logger.info("Tokenizer loaded")
    showTokenizer(tokenizer)

    # Load datasets
    train_df = load_csv_as_df(args.train_file, args.extend_vocab)
    valid_df = load_csv_as_df(args.valid_file, args.extend_vocab)
    test_df = load_csv_as_df(args.test_file, args.extend_vocab)
    if "ml-1m" in args.train_file.lower():
        extension_fields = ["User ID", "Movie ID"]
    else:
        extension_fields = []
    tokenizer = vocabulary_extension(
        tokenizer=tokenizer,
        df_list=[train_df, test_df],
        extension_method=args.extend_vocab,
        extension_fields=extension_fields
    )
    logger.info("Tokenizer extended")
    showTokenizer(tokenizer)

    datasets = {
        "train": MyDataset(train_df),
        "valid": MyDataset(valid_df),
        "test": MyDataset(test_df),
    }
    for split_name in ["train", "valid", "test"]:
        datasets[split_name].setup(
            tokenizer=tokenizer,
            mode=split_name,
            args=args
        )
        logger.info(datasets[split_name])
    n_nodes = datasets["train"].n_nodes
    n_relations = datasets["train"].n_relations

    data_collator = MyDataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(datasets["train"], batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(datasets["valid"], batch_size=args.per_device_eval_batch_size, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = MyModel(args, n_nodes, n_relations).to(device)

    # IMPORTANT! Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = args.num_train_epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in progress_bar:
            users = batch['users'].to(device)
            movies = batch['movies'].to(device)
            user_neighbors = batch['user_neighbors'].to(device)
            movie_neighbors = batch['movie_neighbors'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(users, movies, user_neighbors, movie_neighbors, input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'training_loss': total_loss / (progress_bar.n + 1)})

        # Validation loop (optional)
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                users = batch['users'].to(device)
                movies = batch['movies'].to(device)
                user_neighbors = batch['user_neighbors'].to(device)
                movie_neighbors = batch['movie_neighbors'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(users, movies, user_neighbors, movie_neighbors, input_ids, attention_mask)


if __name__ == "__main__":
    main()