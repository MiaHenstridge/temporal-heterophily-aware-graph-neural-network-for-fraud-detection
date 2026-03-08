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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
)

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
parser = argparse.ArgumentParser('TGN Direct Node Classification')
parser.add_argument('-d', '--data',       type=str,   default='dgraphfin')
parser.add_argument('--bs',              type=int,   default=512)
parser.add_argument('--n_epoch',         type=int,   default=50)
parser.add_argument('--lr',              type=float, default=0.0001)
parser.add_argument('--drop_out',        type=float, default=0.1)
parser.add_argument('--gpu',             type=int,   default=0)
parser.add_argument('--n_layer',         type=int,   default=1)
parser.add_argument('--node_dim',        type=int,   default=64)
parser.add_argument('--time_dim',        type=int,   default=64)
parser.add_argument('--memory_dim',      type=int,   default=64)
parser.add_argument('--n_neighbor',      type=int,   default=10)
parser.add_argument('--prefix',          type=str,   default='')
parser.add_argument('--tune',            action='store_true')

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

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-tgn-node-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-tgn-node-{DATA}-{epoch}.pth'

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
labels     = raw['y']

# node features
x = raw['x']   # (3700550, 17)
node_feat_dim = x.shape[1]

# remap labels: class 1 = fraud, everything else = 0
def remap_labels(mask):
    return (labels[mask] == 1).astype(np.float32)

train_labels = remap_labels(train_mask)
val_labels   = remap_labels(val_mask)
test_labels  = remap_labels(test_mask)

# 1-based node indices
train_nodes = train_mask + 1
val_nodes   = val_mask   + 1
test_nodes  = test_mask  + 1

logger.info(f'Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}')
logger.info(f'Train fraud rate: {train_labels.mean():.4f}')

# edge arrays
src_l = torch.tensor(g_df.u.values,   dtype=torch.long)
dst_l = torch.tensor(g_df.i.values,   dtype=torch.long)
ts_l  = torch.tensor(g_df.ts.values,  dtype=torch.long)
msg   = torch.zeros(len(src_l), 1)

max_idx = max(src_l.max().item(), dst_l.max().item())

# build 1-based node feature matrix (row 0 = padding)
node_feats = np.zeros((max_idx + 2, node_feat_dim), dtype=np.float32)
node_feats[1:len(x)+1] = x

# full TemporalData (used to replay memory state before eval)
full_data = TemporalData(src=src_l, dst=dst_l, t=ts_l, msg=msg)

# edge split flags for building separate train/val/test replay loaders
train_arr = torch.zeros(max_idx + 2, dtype=torch.bool)
val_arr   = torch.zeros(max_idx + 2, dtype=torch.bool)
test_arr  = torch.zeros(max_idx + 2, dtype=torch.bool)
train_arr[torch.tensor(train_nodes, dtype=torch.long)] = True
val_arr  [torch.tensor(val_nodes,   dtype=torch.long)] = True
test_arr [torch.tensor(test_nodes,  dtype=torch.long)] = True

train_edge_flag = train_arr[src_l] & train_arr[dst_l]
val_edge_flag   = val_arr[src_l]   & val_arr[dst_l]
test_edge_flag  = test_arr[src_l]  & test_arr[dst_l]

# replay loaders — used to warm up memory state before node eval
# we stream all edges to build memory, then classify nodes
full_replay_loader = TemporalDataLoader(full_data, batch_size=BATCH_SIZE)

logger.info(f'Total edges: {len(src_l)}')
logger.info(f'Train edges: {train_edge_flag.sum()}, Val edges: {val_edge_flag.sum()}, Test edges: {test_edge_flag.sum()}')

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
        rel_t     = (last_update[edge_index[0]] - t).float()
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

# node classifier head
node_clf = torch.nn.Sequential(
    torch.nn.Linear(NODE_DIM, NODE_DIM // 2),
    torch.nn.ReLU(),
    torch.nn.Dropout(DROP_OUT),
    torch.nn.Linear(NODE_DIM // 2, 1),
).to(device)

neighbor_loader = LastNeighborLoader(max_idx + 2, size=NUM_NEIGHBOR, device=device)

# node feature projection: 17-dim -> MEMORY_DIM
node_feats_th  = torch.tensor(node_feats, dtype=torch.float).to(device)
node_feat_proj = torch.nn.Linear(node_feat_dim, MEMORY_DIM).to(device)

# pos_weight for class imbalance (fraud rate is ~1%)
n_neg = (train_labels == 0).sum()
n_pos = (train_labels == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    list(tgn_memory.parameters()) + list(gnn.parameters()) +
    list(node_clf.parameters())   + list(node_feat_proj.parameters()),
    lr=LEARNING_RATE
)

################## helper: get node embedding ##################
def get_node_embeddings(nodes_tensor):
    """
    Compute embeddings for a batch of nodes using current memory state.
    nodes_tensor: 1D LongTensor of global node ids (1-based).
    Returns: (N, NODE_DIM) embedding tensor.
    """
    n_id, edge_index, e_id = neighbor_loader(nodes_tensor)

    assoc = torch.full((max_idx + 2,), -1, dtype=torch.long, device=device)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = tgn_memory(n_id)
    z = z + node_feat_proj(node_feats_th[n_id])

    if edge_index.size(1) > 0:
        local_edge_index = assoc[edge_index]
        valid_mask = (local_edge_index[0] >= 0) & (local_edge_index[1] >= 0)
        local_edge_index = local_edge_index[:, valid_mask]
        e_id_valid = e_id[valid_mask]

        if local_edge_index.size(1) > 0:
            # e_id references the global edge list — use ts_l and msg directly
            safe_e_id = e_id_valid % len(ts_l)
            edge_t   = ts_l[safe_e_id].to(device)
            edge_msg = msg[safe_e_id].to(device)
            z = gnn(z, last_update, local_edge_index, edge_t, edge_msg)

    return z[assoc[nodes_tensor]]


################## replay memory: warm up state from edge stream ##################
def replay_memory(edge_flag=None):
    """
    Stream edges through memory to build up node states.
    If edge_flag is provided, only replay those edges.
    """
    tgn_memory.reset_state()
    neighbor_loader.reset_state()

    with torch.no_grad():
        tgn_memory.memory.data = node_feat_proj(node_feats_th)

    if edge_flag is None:
        replay_data = full_data
    else:
        replay_data = full_data[edge_flag]

    loader = TemporalDataLoader(replay_data, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            src, dst, t, m = batch.src, batch.dst, batch.t, batch.msg
            neighbor_loader.insert(src, dst)
            tgn_memory.update_state(src, dst, t.long(), m)
            tgn_memory.detach()


################## eval ##################
@torch.no_grad()
def eval_node_clf(nodes, labels_np, edge_flag_for_replay):
    """
    Evaluate node classification.
    Replays memory up to the relevant edges, then classifies nodes in batches.
    """
    tgn_memory.eval()
    gnn.eval()
    node_clf.eval()

    # warm up memory with the appropriate edge history
    replay_memory(edge_flag=edge_flag_for_replay)

    num_instance = len(nodes)
    all_preds, all_labels = [], []

    for s in range(0, num_instance, BATCH_SIZE):
        e = min(num_instance, s + BATCH_SIZE)
        node_batch  = torch.tensor(nodes[s:e],    dtype=torch.long, device=device)
        label_batch = labels_np[s:e]

        z    = get_node_embeddings(node_batch)
        pred = node_clf(z).squeeze(-1)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label_batch)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    scores     = torch.sigmoid(torch.tensor(all_preds)).numpy()
    pred_label = (scores > 0.5).astype(int)

    auc = roc_auc_score(all_labels, scores)
    ap  = average_precision_score(all_labels, scores)
    f1  = f1_score(all_labels, pred_label, zero_division=0)
    mcc = matthews_corrcoef(all_labels, pred_label)
    rc  = recall_score(all_labels, pred_label, zero_division=0)
    pr  = precision_score(all_labels, pred_label, zero_division=0)

    return auc, ap, f1, mcc, rc, pr


################## training ##################
early_stopper = EarlyStopMonitor()

mlflow.set_experiment('tgn-node-direct')
with mlflow.start_run():
    mlflow.log_params(vars(args))

    # shuffle train node indices once
    train_idx = np.arange(len(train_nodes))

    for epoch in range(NUM_EPOCH):
        tgn_memory.train()
        gnn.train()
        node_clf.train()

        # reset and warm up memory on train edges before each epoch
        replay_memory(edge_flag=train_edge_flag)
        tgn_memory.train()
        gnn.train()
        node_clf.train()

        np.random.shuffle(train_idx)
        m_loss = []
        pbar = tqdm(range(0, len(train_nodes), BATCH_SIZE), desc=f'Epoch {epoch}')

        for s in pbar:
            e = min(len(train_nodes), s + BATCH_SIZE)
            batch_idx   = train_idx[s:e]
            node_batch  = torch.tensor(train_nodes[batch_idx], dtype=torch.long, device=device)
            label_batch = torch.tensor(train_labels[batch_idx], dtype=torch.float, device=device)

            z    = get_node_embeddings(node_batch)
            pred = node_clf(z).squeeze(-1)
            loss = criterion(pred, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tgn_memory.detach()

            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        # validation — replay train+val edges so val nodes have full context
        val_replay_flag = train_edge_flag | val_edge_flag
        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr = eval_node_clf(
            val_nodes, val_labels, val_replay_flag
        )

        logger.info(
            f'Epoch {epoch} | loss: {np.mean(m_loss):.4f} | '
            f'val auc: {val_auc:.4f} | val ap: {val_ap:.4f} | '
            f'val f1: {val_f1:.4f} | val mcc: {val_mcc:.4f} | '
            f'val recall: {val_rc:.4f} | val precision: {val_pr:.4f}'
        )
        mlflow.log_metrics({
            'loss':          np.mean(m_loss),
            'val_auc':       val_auc,
            'val_ap':        val_ap,
            'val_f1':        val_f1,
            'val_mcc':       val_mcc,
            'val_recall':    val_rc,
            'val_precision': val_pr,
        }, step=epoch)

        if early_stopper.early_stop_check(val_auc):
            logger.info(f'Early stopping at epoch {epoch}')
            tgn_memory.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
            gnn.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.gnn'))
            node_clf.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.clf'))
            break
        else:
            torch.save(tgn_memory.state_dict(), get_checkpoint_path(epoch))
            torch.save(gnn.state_dict(),         get_checkpoint_path(epoch) + '.gnn')
            torch.save(node_clf.state_dict(),    get_checkpoint_path(epoch) + '.clf')

    # final test — replay all edges for full context
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr = eval_node_clf(
        test_nodes, test_labels, edge_flag_for_replay=None
    )
    logger.info(
        f'Test | auc: {test_auc:.4f} | ap: {test_ap:.4f} | '
        f'f1: {test_f1:.4f} | mcc: {test_mcc:.4f} | '
        f'recall: {test_rc:.4f} | precision: {test_pr:.4f}'
    )
    mlflow.log_metrics({
        'test_auc':       test_auc,
        'test_ap':        test_ap,
        'test_f1':        test_f1,
        'test_mcc':       test_mcc,
        'test_recall':    test_rc,
        'test_precision': test_pr,
    })

    torch.save(tgn_memory.state_dict(), MODEL_SAVE_PATH)
    torch.save(gnn.state_dict(),         MODEL_SAVE_PATH + '.gnn')
    torch.save(node_clf.state_dict(),    MODEL_SAVE_PATH + '.clf')
    mlflow.pytorch.log_model(tgn_memory, 'tgn_memory')
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')