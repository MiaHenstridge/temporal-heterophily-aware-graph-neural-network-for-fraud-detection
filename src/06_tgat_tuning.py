"""
baseline_tgat.py
================
TGAT (Temporal Graph Attention) baseline for DGraphFin node classification.

Architecture
------------
Implements the TGAT model from:
    "Inductive Representation Learning on Temporal Graphs" (Xu et al., ICLR 2020)

Key ideas reproduced faithfully from the reference repo (hxttkl/DGraph_Experiments):
  - Sinusoidal / Bochner time encoding: maps scalar Δt → R^time_dim via
      φ(t) = [cos(ω_0·t), cos(ω_1·t), ..., cos(ω_{d-1}·t)]
    where ω_k = 1 / 10000^(2k/d)  (same schedule as Transformer positional enc.)
  - Node time = min out-edge timestamp for each node  (DGraph paper §4.3)
  - Edge temporal feature = φ(|t_edge − t_src|) passed as edge_attr to
    PyG's TransformerConv (edge_dim pathway)
  - Each conv layer: TransformerConv(in + time_dim → hidden, heads, edge_dim=time_dim)
    followed by BN + ReLU + Dropout
  - MLP classifier head identical to the static GNN baselines

Data
----
Reuses the existing DGraphFin InMemoryDataset (caches processed/data.pt).
A new helper  load_dgraphfin_temporal()  extends load_dgraphfin() by:
  1. Computing node_time  (min out-edge timestamp per node)
  2. Attaching edge_time and node_time to the graph Data object
  3. NOT symmetrising the adjacency — temporal edges are directed by definition;
     we add reverse edges with reversed time deltas so NeighborLoader still sees
     bidirectional neighbourhoods.

PyG usage
---------
  TransformerConv   – multi-head dot-product attention with edge features
  NeighborLoader    – mini-batch subgraph sampling (same as static baselines)
  InMemoryDataset   – dataset caching (shared with static baselines)

Run
---
    python src/baseline_tgat.py --n_epoch 200 --bs 1024 --prefix run1
"""

import os
import sys
import time
import logging
import argparse
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

from torch_geometric.loader import NeighborLoader
from models import TGATModel

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import EarlyStopMonitor
from namespaces import DA
from dgraphfin import DGraphFin, load_dgraphfin_temporal  # reuse existing dataset class

# ── working directory: project root ──────────────────────────────────────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log,         exist_ok=True)
os.makedirs('./saved_models',     exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser('TGAT Node Classification on DGraphFin')
parser.add_argument('-d', '--data',      type=str,   default='DGraphFin')
parser.add_argument('--data_dir',        type=str,   default='./datasets')
parser.add_argument('--bs',              type=int,   default=1024)
parser.add_argument('--n_epoch',         type=int,   default=200)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--drop_out',        type=float, default=0.2)
parser.add_argument('--gpu',             type=int,   default=0)
parser.add_argument('--n_layer',         type=int,   default=2)
parser.add_argument('--node_dim',        type=int,   default=128,
                    help='hidden dimension (must be divisible by n_head)')
parser.add_argument('--time_dim',        type=int,   default=100,
                    help='dimension of the sinusoidal time encoding')
parser.add_argument('--n_head',          type=int,   default=2,
                    help='number of attention heads in TransformerConv')
parser.add_argument('--n_neighbor',      type=int,   default=10,
                    help='neighbors sampled per layer in NeighborLoader')
parser.add_argument('--temporal_strategy', type=str, default='uniform')
parser.add_argument('--fold',            type=int,   default=0)
parser.add_argument('--prefix',          type=str,   default='')
parser.add_argument('--max_round',       type=int,   default=10)
parser.add_argument('--tolerance',       type=float, default=1e-4)
parser.add_argument('--pos_weight',      type=float, default=100,
                    help='positive class weight; -1 = auto (n_neg/n_pos)')
parser.add_argument('--num_workers',     type=int,   default=12)
parser.add_argument('--weight_decay',    type=float, default=5e-7)
parser.add_argument('--to_undirected',  action='store_true',   default=False,
                    help='symmetrize edges into undirected edges')
parser.add_argument('--early_stop_higher_better', action='store_true', default=False)

try:
    args = parser.parse_args()
except SystemExit:
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
NUM_LAYER     = args.n_layer
NUM_NEIGHBOR  = args.n_neighbor
TEMPORAL_STRATEGY = args.temporal_strategy
N_HEAD        = args.n_head
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
NUM_WORKERS   = args.num_workers
WEIGHT_DECAY  = args.weight_decay
EARLY_STOP_HIGHER_BETTER = args.early_stop_higher_better

MODEL_SAVE_PATH     = f'./saved_models/tgat-{args.prefix}-node-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/tgat-{args.prefix}-node-{DATA}-{epoch}.pt'

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(DA.paths.log, f'{time.time()}.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt); ch.setFormatter(fmt)
logger.addHandler(fh); logger.addHandler(ch)
logger.info(args)



# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

(graph, train_idx, val_idx, test_idx,
 train_labels_np, node_feat_dim) = load_dgraphfin_temporal(
    data_dir=args.data_dir, fold=args.fold, to_undirected=args.to_undirected,
)

logger.info(
    f'Graph: {graph.num_nodes:,} nodes | {graph.num_edges:,} edges '
    f'(bidirectional, {graph.num_edges // 2:,} unique)'
)
logger.info(
    f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}'
)
logger.info(f'Train fraud rate: {train_labels_np.mean():.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# NeighborLoaders
# ─────────────────────────────────────────────────────────────────────────────

if NUM_LAYER == 1:
    num_neighbors = [NUM_NEIGHBOR]
elif NUM_LAYER == 2:
    num_neighbors = [NUM_NEIGHBOR, 5]
else:
    step = max(1, (NUM_NEIGHBOR - 5) // (NUM_LAYER - 1))
    num_neighbors = [max(5, NUM_NEIGHBOR - i * step) for i in range(NUM_LAYER)]

logger.info(f'NeighborLoader fanouts (outer→inner): {num_neighbors}')

# NeighborLoader propagates all node/edge attributes automatically because
# graph.edge_attr and graph.node_time are registered as Data attributes.
train_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    temporal_strategy = TEMPORAL_STRATEGY,
    batch_size    = BATCH_SIZE,
    input_nodes   = train_idx,
    input_time    = graph.node_time[train_idx],
    time_attr     = 'edge_time',
    shuffle       = True,
    num_workers   = NUM_WORKERS,
)

val_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    temporal_strategy = TEMPORAL_STRATEGY,
    batch_size    = BATCH_SIZE * 2,
    input_nodes   = val_idx,
    input_time    = graph.node_time[val_idx],
    time_attr     = 'edge_time',
    shuffle       = False,
    num_workers   = NUM_WORKERS,
)

test_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    temporal_strategy = TEMPORAL_STRATEGY,
    batch_size    = BATCH_SIZE * 2,
    input_nodes   = test_idx,
    input_time    = graph.node_time[test_idx],
    time_attr     = 'edge_time',
    shuffle       = False,
    num_workers   = NUM_WORKERS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = TGATModel(
    in_channels     = node_feat_dim,
    hidden_channels = NODE_DIM,
    n_layers        = NUM_LAYER,
    n_head          = N_HEAD,
    time_dim        = TIME_DIM,
    dropout         = DROP_OUT,
).to(device)

logger.info(
    f'TGAT | layers: {NUM_LAYER} | hidden: {NODE_DIM} | '
    f'heads: {N_HEAD} | time_dim: {TIME_DIM} | '
    f'params: {sum(p.numel() for p in model.parameters()):,}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Loss / optimiser
# ─────────────────────────────────────────────────────────────────────────────

n_neg = (train_labels_np == 0).sum()
n_pos = (train_labels_np == 1).sum()
pw    = float(n_neg) / float(n_pos) if args.pos_weight < 0 else args.pos_weight
logger.info(f'pos_weight: {pw:.2f}')

pos_weight = torch.tensor([pw], dtype=torch.float, device=device)
criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_nodes(loader):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch      = batch.to(device, non_blocking=True)
        batch_size = batch.batch_size

        logits = model(
            x          = batch.x,
            edge_index = batch.edge_index,
            time       = batch.edge_time,
            node_time  = batch.node_time,
            batch_size = batch_size,
        )
        all_preds.append(logits.cpu().numpy())
        all_labels.append(batch.y[:batch_size].cpu().numpy())

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

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

early_stopper    = EarlyStopMonitor(max_round=MAX_ROUND, higher_better=EARLY_STOP_HIGHER_BETTER, tolerance=TOLERANCE)
last_saved_epoch = -1

train_loss_hist = []
val_loss_hist   = []
val_auc_hist    = []

mlflow.set_experiment(f'tgat-baseline')
with mlflow.start_run():
    mlflow.log_params(vars(args))

    for epoch in range(NUM_EPOCH):
        model.train()
        m_loss = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            batch      = batch.to(device, non_blocking=True)
            batch_size = batch.batch_size

            logits = model(
                x          = batch.x,
                edge_index = batch.edge_index,
                time       = batch.edge_time,
                node_time  = batch.node_time,
                batch_size = batch_size,
            )
            loss = criterion(logits, batch.y[:batch_size])
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
            model.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
            break
        else:
            if early_stopper.best_epoch == epoch:
                prev = get_checkpoint_path(last_saved_epoch)
                if os.path.exists(prev):
                    os.remove(prev)
                torch.save(model.state_dict(), get_checkpoint_path(epoch))
                last_saved_epoch = epoch

    # ── final test ───────────────────────────────────────────────────────────
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, _ = eval_nodes(test_loader)
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

    # ── training curve ───────────────────────────────────────────────────────
    epochs_list = list(range(len(train_loss_hist)))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss_train = '#e05c5c'
    color_loss_val   = '#e09c5c'
    color_auc        = '#5c8de0'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color_loss_train)
    ax1.plot(epochs_list, train_loss_hist, color=color_loss_train, linewidth=2, label='Train Loss')
    ax1.plot(epochs_list, val_loss_hist,   color=color_loss_val,   linewidth=2, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss_train)

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

    plt.title('TGAT Node Classification — Training Curve (early stop on val loss)')
    plt.tight_layout()

    plot_path = f'./saved_models/{args.prefix}-tgat-node-{DATA}-training-curve.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    mlflow.log_artifact(plot_path)
    logger.info(f'Training curve saved to {plot_path}')