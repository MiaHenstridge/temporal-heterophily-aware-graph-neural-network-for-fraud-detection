import os
import math
import time
import logging
import argparse
import sys

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from torch_geometric.nn import TransformerConv

import mlflow
import mlflow.pytorch
from utils import EarlyStopMonitor
from namespaces import DA

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log, exist_ok=True)
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

################## argument parser ##################
parser = argparse.ArgumentParser('TGN Stage 1: Link Prediction')
parser.add_argument('-d', '--data', type=str, default='dgraphfin')
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--node_dim', type=int, default=64)
parser.add_argument('--time_dim', type=int, default=64)
parser.add_argument('--memory_dim', type=int, default=64)
parser.add_argument('--n_neighbor', type=int, default=10, help='number of neighbors to sample')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--tune', action='store_true')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE    = args.bs
NUM_EPOCH     = args.n_epoch
LEARNING_RATE = args.lr
DROP_OUT      = args.drop_out
GPU           = args.gpu
DATA          = args.data
NODE_DIM      = args.node_dim
TIME_DIM      = args.time_dim
MEMORY_DIM    = args.memory_dim
NUM_NEIGHBOR  = args.n_neighbor
NUM_LAYER     = args.n_layer

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-tgn-edge-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-tgn-edge-{DATA}-{epoch}.pth'

################## logger ##################
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

################## load data ##################
g_df = pd.read_csv(os.path.join(DA.paths.output_data_graph, f'ml_{DATA}.csv'))
raw  = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))

train_mask = raw['train_mask']
val_mask   = raw['valid_mask']
test_mask  = raw['test_mask']

# node features
x = raw['x']  # (3700550, 17)
node_feat_dim = x.shape[1]

src_l = torch.tensor(g_df.u.values, dtype=torch.long)
dst_l = torch.tensor(g_df.i.values, dtype=torch.long)
ts_l  = torch.tensor(g_df.ts.values, dtype=torch.long)
msg   = torch.zeros(len(src_l), 1)   # no edge features

data = TemporalData(src=src_l, dst=dst_l, t=ts_l, msg=msg)

max_idx = max(src_l.max().item(), dst_l.max().item())

# build 1-based node feature matrix (row 0 = padding zeros)
node_feats = np.zeros((max_idx + 2, node_feat_dim), dtype=np.float32)
node_feats[1:len(x)+1] = x

# vectorized edge flags from node masks
train_nodes = torch.tensor(train_mask + 1, dtype=torch.long)
val_nodes   = torch.tensor(val_mask   + 1, dtype=torch.long)
test_nodes  = torch.tensor(test_mask  + 1, dtype=torch.long)

train_arr = torch.zeros(max_idx + 2, dtype=torch.bool)
val_arr   = torch.zeros(max_idx + 2, dtype=torch.bool)
test_arr  = torch.zeros(max_idx + 2, dtype=torch.bool)
train_arr[train_nodes] = True
val_arr[val_nodes]     = True
test_arr[test_nodes]   = True

train_flag = train_arr[src_l] & train_arr[dst_l]
val_flag   = val_arr[src_l]   & val_arr[dst_l]
test_flag  = test_arr[src_l]  & test_arr[dst_l]

if args.tune:
    train_data = data[train_flag]
    val_data   = data[val_flag]
    test_data  = data[val_flag]
else:
    train_data = data[train_flag | val_flag]
    val_data   = data[val_flag]
    test_data  = data[test_flag]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader   = TemporalDataLoader(val_data,   batch_size=BATCH_SIZE)
test_loader  = TemporalDataLoader(test_data,  batch_size=BATCH_SIZE)

logger.info(f'Train edges: {train_flag.sum()}, Val edges: {val_flag.sum()}, Test edges: {test_flag.sum()}')

################## model ##################
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

tgn_memory = TGNMemory(
    num_nodes=max_idx + 2,
    raw_msg_dim=1,
    memory_dim=MEMORY_DIM,
    time_dim=TIME_DIM,
    message_module=IdentityMessage(1, MEMORY_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

class GraphAttnEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.time_enc = tgn_memory.time_enc
        self.convs = torch.nn.ModuleList([
            TransformerConv(in_channels, out_channels // 4, heads=4,
                           dropout=dropout, edge_dim=msg_dim + time_dim)
            for _ in range(n_layers)
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(out_channels) for _ in range(n_layers)
        ])

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t     = (last_update[edge_index[0]] - t).float()  # cast to float for time encoder
        rel_t_enc = self.time_enc(rel_t.unsqueeze(-1))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_attr).relu())
        return x

gnn = GraphAttnEmbedding(
    in_channels=MEMORY_DIM,
    out_channels=NODE_DIM,
    msg_dim=1,
    time_dim=TIME_DIM,
    n_layers=NUM_LAYER,
    dropout=DROP_OUT,
).to(device)

# link prediction head
link_pred = torch.nn.Sequential(
    torch.nn.Linear(NODE_DIM * 2, NODE_DIM),
    torch.nn.ReLU(),
    torch.nn.Dropout(DROP_OUT),
    torch.nn.Linear(NODE_DIM, 1),
).to(device)

neighbor_loader = LastNeighborLoader(max_idx + 2, size=NUM_NEIGHBOR, device=device)

# node feature projection: projects 17-dim node features into memory space
node_feats_th  = torch.tensor(node_feats, dtype=torch.float).to(device)
node_feat_proj = torch.nn.Linear(node_feat_dim, MEMORY_DIM).to(device)

optimizer = torch.optim.Adam(
    list(tgn_memory.parameters()) + list(gnn.parameters()) +
    list(link_pred.parameters()) + list(node_feat_proj.parameters()),
    lr=LEARNING_RATE
)
criterion = torch.nn.BCEWithLogitsLoss()


################## helper: safe subgraph forward ##################
def get_embeddings(src, dst, neg_dst, t, msg, batch_msg_len):
    """
    Safely compute node embeddings for src, dst, neg_dst.
    Handles out-of-bounds assoc indices by filtering invalid edges.
    """
    all_nodes = torch.cat([src, dst, neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(all_nodes)

    # map global node ids -> local subgraph indices
    assoc = torch.full((max_idx + 2,), -1, dtype=torch.long, device=device)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = tgn_memory(n_id)

    # augment memory with projected node features (residual addition)
    z = z + node_feat_proj(node_feats_th[n_id])

    # filter out edges where either endpoint is not in assoc
    local_edge_index = assoc[edge_index]
    valid_mask = (local_edge_index[0] >= 0) & (local_edge_index[1] >= 0)
    local_edge_index = local_edge_index[:, valid_mask]
    e_id_valid = e_id[valid_mask]

    # e_id can reference historical edges outside current batch — clamp safely
    safe_e_id = e_id_valid % batch_msg_len
    edge_t   = t[safe_e_id]
    edge_msg = msg[safe_e_id]

    z = gnn(z, last_update, local_edge_index, edge_t, edge_msg)
    return z, assoc


################## eval ##################
@torch.no_grad()
def eval_link_pred(loader):
    tgn_memory.eval()
    gnn.eval()
    link_pred.eval()

    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(1, max_idx + 1, (src.size(0),), device=device)

        neighbor_loader.insert(src, dst)
        z, assoc = get_embeddings(src, dst, neg_dst, t, msg, len(t))

        pos_out = link_pred(torch.cat([z[assoc[src]], z[assoc[dst]]], dim=-1))
        neg_out = link_pred(torch.cat([z[assoc[src]], z[assoc[neg_dst]]], dim=-1))

        pred  = torch.cat([pos_out, neg_out]).squeeze().cpu()
        label = torch.cat([torch.ones(src.size(0)), torch.zeros(src.size(0))])

        all_preds.append(pred.numpy())
        all_labels.append(label.numpy())

        tgn_memory.update_state(src, dst, t.long(), msg)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, torch.sigmoid(torch.tensor(all_preds)).numpy())
    ap  = average_precision_score(all_labels, torch.sigmoid(torch.tensor(all_preds)).numpy())
    return auc, ap


################## training ##################
early_stopper = EarlyStopMonitor()

mlflow.set_experiment('tgn-edge')
with mlflow.start_run():
    mlflow.log_params(vars(args))

    for epoch in range(NUM_EPOCH):
        tgn_memory.train()
        gnn.train()
        link_pred.train()
        tgn_memory.reset_state()
        neighbor_loader.reset_state()

        # initialize memory with projected node features
        with torch.no_grad():
            tgn_memory.memory.data = node_feat_proj(node_feats_th)

        m_loss = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            batch = batch.to(device)
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

            neg_dst = torch.randint(1, max_idx + 1, (src.size(0),), device=device)

            neighbor_loader.insert(src, dst)
            z, assoc = get_embeddings(src, dst, neg_dst, t, msg, len(t))

            pos_out = link_pred(torch.cat([z[assoc[src]], z[assoc[dst]]], dim=-1))
            neg_out = link_pred(torch.cat([z[assoc[src]], z[assoc[neg_dst]]], dim=-1))

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            tgn_memory.update_state(src, dst, t.long(), msg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tgn_memory.detach()

            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        val_auc, val_ap = eval_link_pred(val_loader)
        logger.info(f'Epoch {epoch} | loss: {np.mean(m_loss):.4f} | val auc: {val_auc:.4f} | val ap: {val_ap:.4f}')
        mlflow.log_metrics({'loss': np.mean(m_loss), 'val_auc': val_auc, 'val_ap': val_ap}, step=epoch)

        if early_stopper.early_stop_check(val_ap):
            logger.info(f'Early stopping at epoch {epoch}')
            tgn_memory.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
            gnn.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.gnn'))
            break
        else:
            torch.save(tgn_memory.state_dict(), get_checkpoint_path(epoch))
            torch.save(gnn.state_dict(), get_checkpoint_path(epoch) + '.gnn')

    test_auc, test_ap = eval_link_pred(test_loader)
    logger.info(f'Test | auc: {test_auc:.4f} | ap: {test_ap:.4f}')
    mlflow.log_metrics({'test_auc': test_auc, 'test_ap': test_ap})

    torch.save(tgn_memory.state_dict(), MODEL_SAVE_PATH)
    torch.save(gnn.state_dict(), MODEL_SAVE_PATH + '.gnn')
    mlflow.pytorch.log_model(tgn_memory, 'tgn_memory')
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')
