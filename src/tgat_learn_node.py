import math
import logging
import time
import sys
import os
import random
import argparse

import tqdm as tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, recall_score, precision_score

from modules import TGAN
from graph_process import NeighborFinder

import mlflow
import mlflow.pytorch
from namespaces import DA

os.chdir('/home/mai/notebooks/final_thesis/')
print(os.getcwd())

os.makedirs(DA.paths.log, exist_ok=True)

class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fxc_2(x))
        x = self.dropout(x)
        return self.fc_3(x)
    
# set random seeds
RANDOM_SEED = 1111
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

######################## argument parser #########################
parser = argparse.ArgumentParser(description="Interface for TGAT experiments on node classification")
parser.add_argument('-d', '--data', type=str, help='data sources to use, try dgraphfin', default='dgraphfin')

parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


###################### logger ######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(os.path.join(DA.paths.log, '{}.log'.format(str(time.time()))))
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.WARN)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


################## load data and train val test split ##################
g_df = pd.read_csv(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}.csv'))
e_feat = np.load(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}.npy'))
n_feat = np.load(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}_node.npy'))

# load original masks and labels from dgraphfin.npz
raw = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))
train_mask = raw['train_mask']
val_mask   = raw['valid_mask']
test_mask  = raw['test_mask']
labels     = raw['y']

# remap labels to 0 and 1
def remap_labels(node_labels):
    return (node_labels == 1).astype(int)

train_labels = remap_labels(labels[train_mask])
val_labels   = remap_labels(labels[val_mask])
test_labels  = remap_labels(labels[test_mask])

# convert 0-based node indices to 1-based (TGAT convention)
train_nodes = train_mask + 1
val_nodes   = val_mask + 1
test_nodes  = test_mask + 1

# full edge arrays
src_l = g_df['u'].values
dst_l = g_df['i'].values
ts_l = g_df['ts'].values
e_idx_l = g_df['idx'].values
label_l = g_df['label'].values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
total_node_set = set(np.unique(np.hstack([src_l, dst_l])))

# train/val/test node sets
train_node_set = set(train_nodes)
val_node_set   = set(val_nodes)
test_node_set  = set(test_nodes)

# flags for training, validation, and test edges
train_flag = np.array([u in train_node_set and i in train_node_set for u, i in zip(src_l, dst_l)])
val_flag = np.array([u in val_node_set and i in val_node_set for u, i in zip(src_l, dst_l)])
test_flag = np.array([u in test_node_set and i in test_node_set for u, i in zip(src_l, dst_l)])


# build train/val/test edge sets 
if args.tune:
    train_src_l = src_l[train_flag]
    train_dst_l = dst_l[train_flag]
    train_ts_l = ts_l[train_flag]
    train_e_idx_l = e_idx_l[train_flag]
    train_label_l = label_l[train_flag]
    # use the validation as test dataset
    test_src_l = src_l[val_flag]
    test_dst_l = dst_l[val_flag]
    test_ts_l = ts_l[val_flag]
    test_e_idx_l = e_idx_l[val_flag]
    test_label_l = label_l[val_flag]
else:
    logger.info('Training use all train data')
    train_src_l = src_l[train_flag | val_flag]
    train_dst_l = dst_l[train_flag | val_flag]
    train_ts_l = ts_l[train_flag | val_flag]
    train_e_idx_l = e_idx_l[train_flag | val_flag]
    train_label_l = label_l[train_flag | val_flag]
    # test on true test dataset
    test_src_l = src_l[test_flag]
    test_dst_l = dst_l[test_flag]
    test_ts_l = ts_l[test_flag]
    test_e_idx_l = e_idx_l[test_flag]
    test_label_l = label_l[test_flag]

##### intialize data structure for graph and edge sampling #####
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))  # undirected

train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))  # undirected

full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

