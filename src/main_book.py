import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train


parser = argparse.ArgumentParser()

#music
# parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, paper, movie)')
# parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=1024, help='cf batch size')
# parser.add_argument('--inverse_r', type=bool, default=True, help='inverse relation in kg')
# parser.add_argument('--n_layer', type=int, default=3, help='the depth of layer')
# parser.add_argument('--n_factor', type=int, default=2, help='the number of factors')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
# parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
# parser.add_argument('--neighbor_size', type=int, default=64, help='the number of triplets in triplet set')
# parser.add_argument('--agg', type=str, default='sum', help='the type of aggregator (sum, decay_sum, pool, concat)')
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=True, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=True, help='whether using random seed or not')

#book
parser.add_argument('-d', '--dataset', type=str, default='book', help='which dataset to use (music, book, paper, movie)')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='cf batch size')
parser.add_argument('--inverse_r', type=bool, default=True, help='inverse relation in kg')
parser.add_argument('--n_layer', type=int, default=2, help='the depth of layer')
parser.add_argument('--n_factor', type=int, default=2, help='the number of factors')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--neighbor_size', type=int, default=64, help='the number of triplets in triplet set')
parser.add_argument('--agg', type=str, default='sum', help='the type of aggregator (sum, pool, concat)')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=True, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

#paper
# parser.add_argument('-d', '--dataset', type=str, default='paper', help='which dataset to use (music, book, paper, movie)')
# parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=1024, help='cf batch size')
# parser.add_argument('--inverse_r', type=bool, default=True, help='inverse relation in kg')
# parser.add_argument('--n_layer', type=int, default=3, help='the depth of layer')
# parser.add_argument('--n_factor', type=int, default=2, help='the number of factors')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
# parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
# parser.add_argument('--neighbor_size', type=int, default=64, help='the number of triplets in triplet set')
# parser.add_argument('--agg', type=str, default='sum', help='the type of aggregator (sum, decay_sum, pool, concat)')
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=True, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=True, help='whether using random seed or not')

# #movie
# parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to use (music, book, paper, movie)')
# parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=128, help='cf batch size')
# parser.add_argument('--inverse_r', type=bool, default=True, help='inverse relation in kg')
# parser.add_argument('--n_layer', type=int, default=2, help='the depth of layer')
# parser.add_argument('--n_factor', type=int, default=2, help='the number of factors')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
# parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
# parser.add_argument('--neighbor_size', type=int, default=64, help='the number of triplets in triplet set')
# parser.add_argument('--agg', type=str, default='decay_sum', help='the type of aggregator (sum, pool, concat)')
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=True, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()

def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(304, 2021)

print(args)    
data_info = load_data(args)
train(args, data_info)
    