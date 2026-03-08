import os
import time
import logging
import argparse
import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
)

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import EarlyStopMonitor
from namespaces import DA

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log, exist_ok=True)
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

################## argument parser ##################
parser = argparse.ArgumentParser('Static GNN Node Classification (GraphSAGE / GAT)')
parser.add_argument('-d', '--data',      type=str,   default='dgraphfin')
parser.add_argument('--model',           type=str,   default='sage', choices=['sage', 'gat'],
                    help='sage = GraphSAGE, gat = GAT')
parser.add_argument('--bs',              type=int,   default=1024,
                    help='number of seed nodes per mini-batch')
parser.add_argument('--n_epoch',         type=int,   default=10)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--drop_out',        type=float, default=0.2)
parser.add_argument('--gpu',             type=int,   default=0)
parser.add_argument('--n_layer',         type=int,   default=2)
parser.add_argument('--node_dim',        type=int,   default=128)
parser.add_argument('--heads',           type=int,   default=4,
                    help='number of attention heads (GAT only)')
parser.add_argument('--n_neighbor',      type=int,   default=10,
                    help='neighbors sampled per layer in NeighborLoader')
parser.add_argument('--prefix',          type=str,   default='')
parser.add_argument('--max_round',       type=int,   default=10)
parser.add_argument('--tolerance',       type=float, default=1e-4)
parser.add_argument('--pos_weight',      type=float, default=-1,
                    help='positive class weight for BCE loss. -1 = auto (n_neg/n_pos)')

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
NUM_LAYER     = args.n_layer
NUM_NEIGHBOR  = args.n_neighbor
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
MODEL_TYPE    = args.model

MODEL_SAVE_PATH     = f'./saved_models/{args.prefix}-{MODEL_TYPE}-node-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{MODEL_TYPE}-node-{DATA}-{epoch}.pth'

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
raw = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))

train_mask = raw['train_mask']
val_mask   = raw['valid_mask']
test_mask  = raw['test_mask']
labels     = raw['y']
x          = raw['x']           # (3700550, 17)
edge_index = raw['edge_index']  # shape may be (N,2) or (2,N)

node_feat_dim = x.shape[1]

# remap labels: class 1 = fraud, everything else = 0
def remap_labels(mask):
    return (labels[mask] == 1).astype(np.float32)

train_labels_base = remap_labels(train_mask)
val_labels        = remap_labels(val_mask)
test_labels       = remap_labels(test_mask)

# ── Node splitting logic (identical to tgn_learn_node.py) ──────────────────
# background nodes (class 2 & 3) included in train, labelled 0
background_mask   = np.where((labels == 2) | (labels == 3))[0]
background_nodes  = background_mask + 1          # 1-based
background_labels = np.zeros(len(background_nodes), dtype=np.float32)

# 1-based node indices
train_nodes_base = train_mask + 1
val_nodes        = val_mask   + 1
test_nodes       = test_mask  + 1

# train includes background nodes
train_nodes  = np.concatenate([train_nodes_base, background_nodes])
train_labels = np.concatenate([train_labels_base, background_labels])

logger.info(f'Train nodes (incl background): {len(train_nodes)}')
logger.info(f'Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}')
logger.info(f'Train fraud rate: {(train_labels == 1).mean():.4f}')

# ── Build PyG graph ───────────────────────────────────────────────────────
# DGraphFin edge_index may be (N,2) — transpose to (2,N)
ei = torch.tensor(edge_index, dtype=torch.long)
if ei.shape[0] != 2:
    ei = ei.t().contiguous()

# build 1-based node feature matrix (row 0 = padding/dummy)
max_node_idx  = int(ei.max().item())
num_nodes     = max_node_idx + 2
node_feats    = np.zeros((num_nodes, node_feat_dim), dtype=np.float32)
node_feats[1:len(x)+1] = x
node_feats_th = torch.tensor(node_feats, dtype=torch.float)

# undirected graph: add reverse edges
ei_full = torch.cat([ei, ei.flip(0)], dim=1)

# full-graph label tensor (-1 = unlabelled) — used inside NeighborLoader batches
y_full = torch.full((num_nodes,), -1.0, dtype=torch.float)
y_full[torch.tensor(train_nodes, dtype=torch.long)] = torch.tensor(train_labels, dtype=torch.float)
y_full[torch.tensor(val_nodes,   dtype=torch.long)] = torch.tensor(val_labels,   dtype=torch.float)
y_full[torch.tensor(test_nodes,  dtype=torch.long)] = torch.tensor(test_labels,  dtype=torch.float)

# graph stays on CPU — NeighborLoader transfers each mini-batch to GPU
graph = Data(x=node_feats_th, edge_index=ei_full, y=y_full, num_nodes=num_nodes)

logger.info(f'Graph: {graph.num_nodes} nodes, {graph.num_edges} edges')

################## device ##################
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

################## NeighborLoaders ##################
# num_neighbors: sample NUM_NEIGHBOR neighbors per hop, listed outer->inner
# e.g. n_layer=2, n_neighbor=10 -> sample 10 neighbors at hop-2, then 10 at hop-1
num_neighbors = [NUM_NEIGHBOR] * NUM_LAYER

train_loader = NeighborLoader(
    graph,
    num_neighbors=num_neighbors,
    batch_size=BATCH_SIZE,
    input_nodes=torch.tensor(train_nodes, dtype=torch.long),
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = NeighborLoader(
    graph,
    num_neighbors=num_neighbors,
    batch_size=BATCH_SIZE * 2,      # no grad — can use larger batch
    input_nodes=torch.tensor(val_nodes, dtype=torch.long),
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

test_loader = NeighborLoader(
    graph,
    num_neighbors=num_neighbors,
    batch_size=BATCH_SIZE * 2,
    input_nodes=torch.tensor(test_nodes, dtype=torch.long),
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

################## model ##################
# In NeighborLoader mini-batch training, the subgraph returned for each batch
# contains seed nodes + their sampled neighbors. Seed nodes are always placed
# first (indices 0..batch_size-1). We run the full GNN on the subgraph and
# only classify the seed nodes at the end.

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, dropout):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # skip connection: concat GNN output with raw input features
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch_size):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
        # slice to seed nodes only, then concat raw features
        return self.clf(torch.cat([h[:batch_size], x[:batch_size]], dim=-1)).squeeze(-1)


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, heads, dropout):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATConv(in_channels, hidden_channels // heads,
                                   heads=heads, dropout=dropout, concat=True))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                       heads=heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch_size):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.clf(torch.cat([h[:batch_size], x[:batch_size]], dim=-1)).squeeze(-1)


if MODEL_TYPE == 'sage':
    model = GraphSAGEModel(
        in_channels=node_feat_dim,
        hidden_channels=NODE_DIM,
        n_layers=NUM_LAYER,
        dropout=DROP_OUT,
    ).to(device)
else:
    model = GATModel(
        in_channels=node_feat_dim,
        hidden_channels=NODE_DIM,
        n_layers=NUM_LAYER,
        heads=args.heads,
        dropout=DROP_OUT,
    ).to(device)

logger.info(f'Model: {MODEL_TYPE.upper()} | params: {sum(p.numel() for p in model.parameters()):,}')

################## loss / optimiser ##################
n_neg = (train_labels == 0).sum()
n_pos = (train_labels == 1).sum()
pw    = float(n_neg) / float(n_pos) if args.pos_weight < 0 else args.pos_weight
logger.info(f'pos_weight: {pw:.2f}')

pos_weight = torch.tensor([pw], dtype=torch.float, device=device)
criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
)

################## eval ##################
@torch.no_grad()
def eval_nodes(loader):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch      = batch.to(device, non_blocking=True)
        batch_size = batch.batch_size   # seed nodes = first batch_size rows

        pred         = model(batch.x, batch.edge_index, batch_size)
        labels_batch = batch.y[:batch_size]

        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

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

    val_loss = torch.nn.BCEWithLogitsLoss()(
        torch.tensor(all_preds), torch.tensor(all_labels)
    ).item()

    return auc, ap, f1, mcc, rc, pr, val_loss

################## training ##################
early_stopper    = EarlyStopMonitor(max_round=MAX_ROUND, higher_better=True, tolerance=TOLERANCE)
last_saved_epoch = -1

train_loss_hist = []
val_loss_hist = []
val_auc_hist    = []

mlflow.set_experiment('static-gnn-node')
with mlflow.start_run():
    mlflow.log_params(vars(args))

    for epoch in range(NUM_EPOCH):
        model.train()
        m_loss = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            batch      = batch.to(device, non_blocking=True)
            batch_size = batch.batch_size   # seed nodes are always the first batch_size rows

            pred        = model(batch.x, batch.edge_index, batch_size)
            label_batch = batch.y[:batch_size]

            loss = criterion(pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr, val_loss = eval_nodes(val_loader)
        scheduler.step(val_auc)

        logger.info(
            f'Epoch {epoch} | loss: {np.mean(m_loss):.4f} | val_loss: {val_loss:.4f} | '
            f'val auc: {val_auc:.4f} | val ap: {val_ap:.4f} | '
            f'val f1: {val_f1:.4f} | val mcc: {val_mcc:.4f} | '
            f'val recall: {val_rc:.4f} | val precision: {val_pr:.4f}'
        )
        mlflow.log_metrics({
            'loss':          np.mean(m_loss),
            'val_loss':      val_loss,
            'val_auc':       val_auc,
            'val_ap':        val_ap,
            'val_f1':        val_f1,
            'val_mcc':       val_mcc,
            'val_recall':    val_rc,
            'val_precision': val_pr,
        }, step=epoch)

        train_loss_hist.append(np.mean(m_loss))
        val_loss_hist.append(val_loss)
        val_auc_hist.append(val_auc)

        if early_stopper.early_stop_check(val_auc):
            logger.info(f'Early stopping at epoch {epoch}')
            model.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.pt'))
            break
        else:
            if early_stopper.best_epoch == epoch:
                prev = get_checkpoint_path(last_saved_epoch) + '.pt'
                if os.path.exists(prev):
                    os.remove(prev)
                torch.save(model.state_dict(), get_checkpoint_path(epoch) + '.pt')
                last_saved_epoch = epoch

    # final test
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, test_loss = eval_nodes(test_loader)
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

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')

    # plot
    epochs_list = list(range(len(train_loss_hist)))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss_train = '#e05c5c'
    color_loss_val   = '#e09c5c'
    color_auc  = '#5c8de0'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color_loss)
    ax1.plot(epochs_list, train_loss_hist, color=color_loss, linewidth=2, label='Train Loss')
    ax1.plot(epochs_list, val_loss_hist, color=color_loss, linewidth=2, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val AUC', color=color_auc)
    ax2.plot(epochs_list, val_auc_hist, color=color_auc, linewidth=2, linestyle='--', label='Val AUC')
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.set_ylim(0, 1)

    best_ep = early_stopper.best_epoch
    ax2.axvline(x=best_ep, color='gray', linestyle=':', linewidth=1.5, label=f'Best epoch ({best_ep})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(f'{MODEL_TYPE.upper()} Node Classification — Training Curve (early stop on val AUC)')
    plt.tight_layout()

    plot_path = f'./saved_models/{args.prefix}-{MODEL_TYPE}-node-{DATA}-training-curve.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    mlflow.log_artifact(plot_path)
    logger.info(f'Training curve saved to {plot_path}')