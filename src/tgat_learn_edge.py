import math
import logging
import time
import random
import sys
import os
import argparse

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from modules import TGAN
from graph_process import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
from namespaces import DA

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

os.makedirs(DA.paths.log, exist_ok=True)
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

# set random seeds
RANDOM_SEED = 1111
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, default='dgraphfin')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### logger
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


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


### Load data
g_df = pd.read_csv(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}.csv'))
e_feat = np.load(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}.npy'))
n_feat = np.load(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}_node.npy'))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

### Train/val/test split using DGraphFin masks
raw = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))
train_mask = raw['train_mask']
val_mask   = raw['valid_mask']
test_mask  = raw['test_mask']

# 1-based node indices
train_nodes = train_mask + 1
val_nodes   = val_mask + 1
test_nodes  = test_mask + 1

# vectorized edge flags
train_node_arr = np.zeros(max_idx + 2, dtype=bool)
val_node_arr   = np.zeros(max_idx + 2, dtype=bool)
test_node_arr  = np.zeros(max_idx + 2, dtype=bool)
train_node_arr[train_nodes] = True
val_node_arr[val_nodes]     = True
test_node_arr[test_nodes]   = True

train_flag = train_node_arr[src_l] & train_node_arr[dst_l]
val_flag   = val_node_arr[src_l]   & val_node_arr[dst_l]
test_flag  = test_node_arr[src_l]  & test_node_arr[dst_l]

train_src_l   = src_l[train_flag]
train_dst_l   = dst_l[train_flag]
train_ts_l    = ts_l[train_flag]
train_e_idx_l = e_idx_l[train_flag]
train_label_l = label_l[train_flag]

val_src_l   = src_l[val_flag]
val_dst_l   = dst_l[val_flag]
val_ts_l    = ts_l[val_flag]
val_e_idx_l = e_idx_l[val_flag]
val_label_l = label_l[val_flag]

test_src_l   = src_l[test_flag]
test_dst_l   = dst_l[test_flag]
test_ts_l    = ts_l[test_flag]
test_e_idx_l = e_idx_l[test_flag]
test_label_l = label_l[test_flag]

logger.info(f'Train edges: {train_flag.sum()}, Val edges: {val_flag.sum()}, Test edges: {test_flag.sum()}')

### Initialize graph data structures
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l.tolist(), train_dst_l.tolist(),
                               train_e_idx_l.tolist(), train_ts_l.tolist()):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l.tolist(), dst_l.tolist(),
                               e_idx_l.tolist(), ts_l.tolist()):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler   = RandEdgeSampler(src_l, dst_l)
test_rand_sampler  = RandEdgeSampler(src_l, dst_l)

### Model initialize
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
tgan = TGAN(
    train_ngh_finder, n_feat, e_feat,
    num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
    seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM
)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()

### Training loop
for epoch in range(NUM_EPOCH):
    tgan.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))

    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut  = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    # validation
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val', tgan, val_rand_sampler,
                                                        val_src_l, val_dst_l, val_ts_l, val_label_l)

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {:.4f}, val acc: {:.4f}'.format(np.mean(acc), val_acc))
    logger.info('train auc: {:.4f}, val auc: {:.4f}'.format(np.mean(auc), val_auc))
    logger.info('train ap: {:.4f},  val ap: {:.4f}'.format(np.mean(ap),  val_ap))

    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        tgan.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

### Test
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test', tgan, test_rand_sampler,
                                                        test_src_l, test_dst_l, test_ts_l, test_label_l)
logger.info('Test -- acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(test_acc, test_auc, test_ap))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN model saved to {}'.format(MODEL_SAVE_PATH))