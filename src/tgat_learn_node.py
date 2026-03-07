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
from utils import EarlyStopMonitor, RandEdgeSampler

import mlflow
import mlflow.pytorch
from namespaces import DA

# os.chdir('/home/mai/notebooks/final_thesis/')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        x = self.act(self.fc_2(x))
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
parser.add_argument('--n_head', type=int, default=3, help="number of attention heads")
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

########## model saving and checkpointing ##########
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'



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


########## model initialization ##########
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
tgan = TGAN(
    ngh_finder=train_ngh_finder,
    n_feat=n_feat,
    e_feat=e_feat,
    num_layers=NUM_LAYER,
    use_time=USE_TIME,
    agg_method=AGG_METHOD,
    attn_mode=ATTN_MODE,
    seq_len=SEQ_LEN,
    n_head=NUM_HEADS,
    drop_out=DROP_OUT,
    node_dim=NODE_DIM,
    time_dim=TIME_DIM,
)
logger.info("Model initialized with {} parameters".format(sum(p.numel() for p in tgan.parameters())))

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
logger.info("Optimizer initialized with learning rate: {}".format(LEARNING_RATE))

# LR classifier on top of TGAN embeddings
lr_model = LR(n_feat.shape[1] if NODE_DIM is None else NODE_DIM).to(device)
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.BCELoss()

tgan = tgan.to(device)
logger.info("Model loaded to device: {}".format(device))

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug("Number of training instances: {}, number of batches per epoch: {}".format(num_instance, num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()

############ train and evaluation loop ############
def eval_node_classification(tgan, lr_model, nodes, labels, ts_l, num_neighbors, device):
    """Evaluate node classification performance."""
    tgan.eval()
    lr_model.eval()
    
    num_instance = len(nodes)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)
            
            node_batch = nodes[s_idx:e_idx]
            label_batch = labels[s_idx:e_idx]
            
            # get cut_time for each node: use the last timestamp in the graph
            # so the node can see all its historical edges
            ts_batch = node_last_ts[node_batch]
            
            node_embed = tgan.tem_conv(node_batch, ts_batch, NODE_LAYER, num_neighbors)
            pred = lr_model(node_embed).sigmoid().squeeze(1)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch)
    
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pred_label = (all_preds > 0.5).astype(int)
    
    auc = roc_auc_score(all_labels, all_preds)
    f1  = f1_score(all_labels, pred_label)
    mcc = matthews_corrcoef(all_labels, pred_label)
    rc  = recall_score(all_labels, pred_label)
    pr  = precision_score(all_labels, pred_label)
    
    return auc, f1, mcc, rc, pr

############ node-level cut times (vectorized) ############
# for each node, find its latest timestamp in the edge list
# used so the model can see all historical edges when generating embeddings
node_last_ts = np.zeros(max_idx + 2, dtype=np.float32)
for n, t in zip(src_l, ts_l):
    if t > node_last_ts[n]:
        node_last_ts[n] = t
for n, t in zip(dst_l, ts_l):
    if t > node_last_ts[n]:
        node_last_ts[n] = t


# train loop
for epoch in range(NUM_EPOCH):
    # training using only training graph
    tgan.ngh_finder = train_ngh_finder
    tgan.train()
    lr_model.train()

    auc, f1, m_loss = [], [], []
    np.random.shuffle(idx_list)
    logger.info("Epoch {}: Starting training...".format(epoch))
    
    for k in range(num_batch):
        s_idx = k*BATCH_SIZE                                # start index of the batch
        e_idx = min(num_instance-1, s_idx + BATCH_SIZE)     # end index of the batch
        batch_idx = idx_list[s_idx:e_idx]                    # indices of the instances in the batch
        
        node_batch = train_nodes[batch_idx]
        label_batch = train_labels[batch_idx]
        ts_batch = node_last_ts[node_batch]

        # get node embeddings from TGAN
        t0 = time.time()
        node_embed = tgan.tem_conv(node_batch, ts_batch, NODE_LAYER, NUM_NEIGHBORS)
        print(f"Batch {k}: tem_conv took {time.time()-t0:.2f}s")

        # classify
        t1 = time.time()
        pred = lr_model(node_embed).sigmoid().squeeze(1)
        label_tensor = torch.tensor(label_batch, dtype=torch.float, device=device)

        loss = criterion(pred, label_tensor)

        optimizer.zero_grad()
        lr_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_optimizer.step()
        print(f"Batch {k}: forward+backward took {time.time()-t1:.2f}s")

        # training metrics
        with torch.no_grad():
            pred_np = pred.cpu().detach().numpy()
            pred_label = (pred_np > 0.5).astype(int)
            m_loss.append(loss.item())
            auc.append(roc_auc_score(label_batch, pred_np))
            f1.append(f1_score(label_batch, pred_label, zero_division=0))

    # validation using full graph data
    tgan.ngh_finder = full_ngh_finder
    val_ts_batch = node_last_ts[val_nodes]
    val_auc, val_f1, val_mcc, val_rc, val_pr = eval_node_classification(
        tgan, lr_model, 
        val_nodes, val_labels, val_ts_batch, 
        NUM_NEIGHBORS, device)

    # logging
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train auc: {:.4f}, val auc: {:.4f}'.format(np.mean(auc), val_auc))
    logger.info('train f1:  {:.4f}, val f1:  {:.4f}'.format(np.mean(f1),  val_f1))
    logger.info('val mcc: {:.4f}, val recall: {:.4f}, val precision: {:.4f}'.format(val_mcc, val_rc, val_pr))

    # early stopping check
    if early_stopper.early_stop_check(val_auc):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        tgan.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
        tgan.eval()
        lr_model.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch)) 


############ final test evaluation ############
tgan.ngh_finder = full_ngh_finder
test_auc, test_f1, test_mcc, test_rc, test_pr = eval_node_classification(
    tgan, lr_model, test_nodes, test_labels, ts_l, NUM_NEIGHBORS, device
)
logger.info('Test auc: {:.4f}, f1: {:.4f}, mcc: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(
    test_auc, test_f1, test_mcc, test_rc, test_pr
))