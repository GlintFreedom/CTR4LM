import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import collections
from typing import Union, Dict, List, Optional, Any, Tuple
import torch
from transformers import DataCollatorWithPadding
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_csv_as_df(file_path, extend_vocab="none"):
    # if 'ml-1m-new' in file_path.lower():
    #     dataset = df_books = pd.read_parquet(file_path)
    #     fields = ['User ID', 'Gender', 'Age', 'Job', 'Zipcode', "Movie ID", "Movie title", "Movie genre", "labels"]
    #     dataset = dataset[fields]
    if 'ml-1m' in file_path.lower():
        dataset = pd.read_csv(file_path, dtype={'Zip-code': 'str'})
        fields = ["User ID", "Gender", "Age", "Occupation", "Zipcode", "Movie ID", "Title", "Film genre", "Label"]
        dataset = dataset[fields]
        if extend_vocab in ["prefix_none", "raw"]:
            dataset['User ID'] = dataset['User ID'].map(lambda x: f'U{x}')
            dataset['Movie ID'] = dataset['Movie ID'].map(lambda x: f'M{x}')
    elif 'bookcrossing' in file_path.lower():
        dataset = pd.read_csv(file_path, dtype={"labels": int, "User ID": str}, sep="\t")
        fields = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', "Author", "Publication year", "Publisher",
                  "labels"]
        dataset = dataset[fields]
    # elif 'az-toys' in file_path.lower():
    #     dataset = pd.read_parquet(file_path)
    #     fields = ["User ID", "Item ID", "Category", "Title", "Brand", "labels"]
    #     dataset = dataset[fields]
    # elif 'goodreads' in file_path.lower():
    #     dataset = df_books = pd.read_parquet(file_path)
    #     fields = ["User ID", "Book ID", "Book title", "Book genres", "Average rating", "Number of book reviews",
    #               "Author ID", "Author name", "Number of pages", "eBook flag", "Format", "Publisher",
    #               "Publication year", "Work ID", "Media type", "labels"]
    #     dataset = dataset[fields]
    else:
        raise NotImplementedError

    return dataset


class MyDataset(Dataset):
    def __init__(self, dataframe):
        super(MyDataset, self).__init__()
        self.data = dataframe
        self.n_nodes = 0
        self.n_relations = 1

    def setup(self, tokenizer, mode, args):
        self.tokenizer = tokenizer
        self.shuffle_fields = args.shuffle_fields
        self.mode = mode
        self.n_layer = args.n_layer
        self.neighbors = self.ckg_propagation(args)

    def ckg_propagation(self, args):
        graph_dict = collections.defaultdict(list)
        neighbor = collections.defaultdict(list)
        graph = self.data[self.data["Label"] == 1][["User ID", "Label", "Movie ID"]]
        users = graph['User ID'].unique()
        items = graph['Movie ID'].unique()
        self.n_nodes = users.shape[0] + items.shape[0]
        users_items = np.concatenate((users, items), axis=0)
        for h, r, t in graph.values:
            # the only one relation with index 0 (relations[0])
            graph_dict[h].append((0, t))
            # IMPORTANT: undirected graph
            graph_dict[t].append((0, h))
        neighbor_size = args.neighbor_size
        for obj in tqdm(users_items, desc=f'{self.mode}: constructing NEIGHBORs of nodes', leave=False):
            for layer in range(args.n_layer):
                h, r, t = [], [], []
                if layer == 0:
                    nodes = [obj]
                else:
                    # get all nodes in the last layer
                    nodes = neighbor[obj][-1][2]
                assert len(nodes) > 0
                for node in nodes:
                    for relation_and_tail in graph_dict[node]:
                        h.append(node)
                        r.append(relation_and_tail[0])
                        t.append(relation_and_tail[1])

                # align the number of neighbors
                while len(h) < neighbor_size:
                    rand_idx = np.random.randint(len(h))
                    h.append(h[rand_idx])
                    r.append(r[rand_idx])
                    t.append(t[rand_idx])

                # restrict the number of neighbors
                indices = np.random.choice(len(h), size=neighbor_size, replace=(len(h) < neighbor_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                neighbor[obj].append((h, r, t))
        logger.info(f"{self.mode}: construct neighbors DONE")
        return neighbor

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"MyDataset object with {len(self)} samples"

    def _get_triple_tensor(self, neighbor_dict):
        h, r, t = [], [], []
        for i in range(self.n_layer):
            h.append(torch.LongTensor([int(j[1:]) for j in neighbor_dict[i][0]]))
            r.append(torch.LongTensor(neighbor_dict[i][1]))
            t.append(torch.LongTensor([int(j[1:]) for j in neighbor_dict[i][2]]))
        h = torch.stack(h)
        r = torch.stack(r)
        t = torch.stack(t)
        return torch.stack([h, r, t])

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        user = item["User ID"]
        movie = item["Movie ID"]
        shuffle_fields = item.index.tolist()
        shuffle_fields.remove("Label")
        label = item["Label"]

        user_neighbors = self.neighbors[user]
        movie_neighbors = self.neighbors[movie]
        user_neighbors = self._get_triple_tensor(user_neighbors)
        movie_neighbors = self._get_triple_tensor(movie_neighbors)

        if self.shuffle_fields:
            random.shuffle(shuffle_fields)

        shuffled_text = " ".join(
            ["%s is %s." % (field, str(item[field]).strip()) for field in shuffle_fields]
        )
        encoding = self.tokenizer(shuffled_text, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=512)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'user': torch.tensor(int(user[1:])),
            'movie': torch.tensor(int(movie[1:])),
            'user_neighbors': user_neighbors,
            'movie_neighbors': movie_neighbors
        }


class MyDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])

        users = torch.stack([f['user'] for f in features])
        movies = torch.stack([f['movie'] for f in features])

        user_neighbors = torch.stack([f['user_neighbors'] for f in features])
        movie_neighbors = torch.stack([f['movie_neighbors'] for f in features])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'users': users,
            'movies': movies,
            'user_neighbors': user_neighbors,
            'movie_neighbors': movie_neighbors
        }
