import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from model import MyModel
import logging
from transformers import AutoTokenizer


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info):
    logging.info("================== training MODEL ====================")
    # train_data, eval_data, test_data, neighbor_dict, n_node, n_relation
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    neighbor_dict = data_info[3]
    user_prompts, item_prompts = data_info[6], data_info[7]

    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    logging.info("Tokenizer loaded")

    model, optimizer, loss_func = _init_model(args, data_info, tokenizer)
    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start:start + args.batch_size, 2])
            input_ids, attention_mask = _get_LM_prompts(args, train_data, start, start + args.batch_size, user_prompts, item_prompts, tokenizer)
            scores = model(*_get_feed_data(args, train_data, neighbor_dict, start, start + args.batch_size), input_ids, attention_mask)
            loss = loss_func(scores, labels)
            # loss = loss_func(scores, labels) + 0.001 * crossloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += args.batch_size
        train_auc, train_f1 = ctr_eval(args, model, train_data, neighbor_dict, user_prompts, item_prompts, tokenizer)
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data, neighbor_dict, user_prompts, item_prompts, tokenizer)
        test_auc, test_f1 = ctr_eval(args, model, test_data, neighbor_dict, user_prompts, item_prompts, tokenizer)
        # wandb.log({"train_auc" : train_auc,"train_f1":train_f1, "eval_auc" : eval_auc,"eval_f1":eval_f1, "test_auc" : test_auc,"test_f1":test_f1})
        # if test_auc > best_auc:
                # print('find a best auc')
                # best_auc = test_auc
                # wandb.run.summary["best_auc"] = best_auc
                # wandb.log({"best_auc" : best_auc})
        
        ctr_info = 'epoch %.2d    train auc: %.4f f1: %.4f    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1)
        # if args.show_topk:
        #     topk_eval(args, model, train_data, test_data, neighbor_dict)


def ctr_eval(args, model, data, neighbor_dict, user_prompts, item_prompts, tokenizer):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        input_ids, attention_mask = _get_LM_prompts(args, data, start, start + args.batch_size, user_prompts,
                                                    item_prompts, tokenizer)
        scores = model(*_get_feed_data(args, data, neighbor_dict, start, start + args.batch_size), input_ids, attention_mask)
        scores = scores.detach().cpu().numpy()  #detach: take tensor form grad
        if(np.nan in scores):
            print(scores)
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except ValueError:
            start += args.batch_size
            continue
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()  
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1


def topk_eval(args, model, train_data, test_data, neighbor_dict):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()    
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size] 
            input_data = _get_topk_feed_data(user, items)
            scores, _ = model(*_get_feed_data(args, input_data, neighbor_dict, 0, args.batch_size))
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores, _ = model(*_get_feed_data(args, input_data, neighbor_dict, 0, args.batch_size))
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
            precision_list[k].append(hit_num / k)
    model.train()
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    _show_recall_info(zip(k_list, recall, precision))


def _init_model(args, data_info, tokenizer):
    n_node = data_info[4]
    n_relation = data_info[5]
    model = MyModel(args, n_node, n_relation, tokenizer)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func


def _get_feed_data(args, data, neighbor_dict, start, end):
    users = torch.LongTensor(data[start:end, 0])
    items = torch.LongTensor(data[start:end, 1])
    if args.use_cuda:
        users = users.cuda()
        items = items.cuda()
    users_triple = _get_triple_tensor(args, data[start:end,0], neighbor_dict)
    items_triple = _get_triple_tensor(args, data[start:end,1], neighbor_dict)
    return users, items, users_triple, items_triple


def _get_feed_label(args, labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels


def _get_LM_prompts(args, data, start, end, user_prompts, item_prompts, tokenizer):
    prompts = []
    end = min(end, data.shape[0])
    for i in range(start, end):
        user = data[i, 0]
        item = data[i, 1]
        prompt = user_prompts[user] + item_prompts[item]
        prompts.append(prompt)

    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    if args.use_cuda:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    return input_ids, attention_mask


def _get_triple_tensor(args, objs, neighbor_dict):
    # [h,r,t]  h: [layers, batch_size, neighbor_size]
    h,r,t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([neighbor_dict[obj][i][0] for obj in objs]))
        r.append(torch.LongTensor([neighbor_dict[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([neighbor_dict[obj][i][2] for obj in objs]))
        if args.use_cuda:
            h = list(map(lambda x: x.cuda(), h))
            r = list(map(lambda x: x.cuda(), r))
            t = list(map(lambda x: x.cuda(), t))
    return [h,r,t]


def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return np.array(res)


def _show_recall_info(recall_zip):
    res = ""
    for i,j,k in recall_zip:
        res += "K@%d:%.4f  %.4f"%(i,j,k)
    logging.info(res)
