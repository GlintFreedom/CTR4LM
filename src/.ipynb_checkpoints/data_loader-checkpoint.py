import collections
import os
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

users = set()
train_users = set()
items = set()
n_user = 0
n_item = 0
n_entity = 0
n_relation = 0
n_node = 0


def load_data(args):
    logging.info("================== preparing data ===================")
    kg = load_kg(args)
    train_data, eval_data, test_data = load_rating(args)
    ckg = construct_ckg(kg, train_data)
    neighbor = ckg_propagation(args, ckg)
    print_info()
    return train_data, eval_data, test_data, neighbor, n_node, n_relation


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_kg_np = kg_np.copy()
        inv_kg_np[:, 0] = kg_np[:, 2]
        inv_kg_np[:, 2] = kg_np[:, 0]
        inv_kg_np[:, 1] = kg_np[:, 1] + max(kg_np[:, 1]) + 1
        # consider additional relation --- 'interact'.
        kg_np[:, 1] = kg_np[:, 1] + 1
        inv_kg_np[:, 1] = inv_kg_np[:, 1] + 1
        # get full version of knowledge graph
        kg = np.concatenate((kg_np, inv_kg_np), axis=0)
    else:
        # consider additional relation --- 'interact'.
        kg_np[:, 1] = kg_np[:, 1] + 1
        kg = kg_np.copy()
    global n_entity, n_relation
    n_entity = max(max(kg[:, 0]), max(kg[:, 2])) + 1
    n_relation = max(kg[:, 1]) + 1
    return kg


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    global n_user, n_node, n_item
    n_user = max(rating_np[:, 0]) + 1
    n_node = n_user + n_entity
    n_item = max(rating_np[:, 1]) + 1
    # remap user id
    rating_np[:, 0] = rating_np[:, 0] + n_entity
    global users, items
    users = set(rating_np[:, 0])
    items = set(rating_np[:, 1])
    return dataset_split(rating_np)


def dataset_split(rating_np):
    logging.info("splitting dataset to 6:2:2 ...")
    train_ratio = 0.6
    eval_ratio = 0.2
    n_rating = rating_np.shape[0]
    train_indices = np.random.choice(n_rating, size=int(n_rating * train_ratio), replace=False)
    train_data = rating_np[train_indices]
    pos_indices = np.where(train_data[:, 2] == 1)
    global train_users
    train_users = set(train_data[pos_indices][:, 0])
    cold_users = set(rating_np[:, 0]) - train_users
    left = set(range(n_rating)) - set(train_indices)
    eval_indices = np.random.choice(list(left), size=int(n_rating * eval_ratio), replace=False)
    eval_data = rating_np[eval_indices]
    test_indices = list(left - set(eval_indices))
    test_data = rating_np[test_indices]
    # remove cold users in data
    for cold_user in cold_users:
        train_indices = np.where(train_data[:, 0] != cold_user)
        eval_indices = np.where(eval_data[:, 0] != cold_user)
        test_indices = np.where(test_data[:, 0] != cold_user)
        train_data = train_data[train_indices]
        eval_data = eval_data[eval_indices]
        test_data = test_data[test_indices]
    return train_data, eval_data, test_data


def construct_ckg(kg, train_data):
    pos_indices = np.where(train_data[:, 2] == 1)
    train_pos_data = train_data[pos_indices]
    cf = train_pos_data.copy()
    cf[:, 2] = cf[:, 1]
    cf[:, 1] = np.array([0] * train_pos_data.shape[0])
    ckg = np.concatenate((kg, cf), axis=0)
    return ckg


def ckg_propagation(args, ckg):
    logging.info("constructing neighbors ...")
    ckg_dict = collections.defaultdict(list)
    neighbor = collections.defaultdict(list)
    for h, r, t in ckg:
         ckg_dict[h].append((t, r))
    user_items = train_users | items
    neighbor_size = args.neighbor_size
    for obj in user_items:
        for layer in range(args.n_layer):
            h, r, t = [], [], []
            if layer == 0:
                nodes = [obj]
            else:
                nodes = neighbor[obj][-1][2]
            for node in nodes:
                for tail_and_relation in ckg_dict[node]:
                    h.append(node)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                indices = np.random.choice(len(h), size=neighbor_size, replace= (len(h) < neighbor_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
            neighbor[obj].append((h, r, t))
    return neighbor


def print_info():
    print('n_user:               %d' % n_user)
    print('n_item:               %d' % n_item)
    print('n_entity:             %d' % n_entity)
    print('n_relation:           %d' % n_relation)
    print('n_node:        %d' % n_node)
