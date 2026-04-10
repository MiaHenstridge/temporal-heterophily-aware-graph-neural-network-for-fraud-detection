"""
07_thegcn_tuning.py
===================
THEGCN training script using the TGL C++ parallel sampler for temporally-
correct neighbourhood sampling.

Prerequisites
-------------
1. Build the C++ sampler extension:
       python src/setup.py build_ext --inplace

2. Prepare the CSC graph data (one-time):
       python src/prepare_sampler_data.py

3. Run training:
       python src/07_thegcn_tuning.py --prefix run1 --n_epoch 50 --bs 1024 \\
           --n_layer 2 --node_dim 128 --time_dim 100 --n_neighbor 10 --gpu 0

Key difference from 06_tgat_tuning.py
--------------------------------------
Uses TemporalSampler (TGL C++ ParallelSampler) instead of PyG NeighborLoader.
Each seed node gets neighbours sampled strictly before its own query timestamp,
which is required for THEGCN's TMP block to learn meaningful (p-q) weights.
The sampler returns SampledBlock objects with precomputed dts (Δt per edge)
that feed directly into TimeEncode.
"""

import os
import sys
import time
import logging
import argparse
import faulthandler
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

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Enable faulthandler so segfaults produce a Python traceback
faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import THEGCNModel, THEGCNSamplerModel
from utils import *
from namespaces import DA
from dgraphfin import load_dgraphfin_temporal
# from sampler import TemporalSampler, load_sampler_data
from sampler_core import ParallelSampler, TemporalGraphBlock

os.makedirs(DA.paths.log,          exist_ok=True)
os.makedirs('./saved_models',      exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser('THEGCN Node Classification on DGraphFin (TGL sampler)')
parser.add_argument('-d', '--data',       type=str,   default='DGraphFin')
parser.add_argument('--data_dir',         type=str,   default='./datasets')
parser.add_argument('--sampler_dir',     type=str,   default='./processed_data/tgl',
                    help='Path to CSC graph .npz built by tgl_data_preprocess.py')
parser.add_argument('--bs',               type=int,   default=1024)
parser.add_argument('--n_epoch',          type=int,   default=100)
parser.add_argument('--lr',               type=float, default=1e-3)
parser.add_argument('--drop_out',         type=float, default=0.2)
parser.add_argument('--gpu',              type=int,   default=0)
parser.add_argument('--n_layer',          type=int,   default=2,
                    help='Number of SMP layers (L in paper). 0 = TMP only.')
parser.add_argument('--node_dim',         type=int,   default=128,
                    help='Hidden dimension for SMP layers and classifier.')
parser.add_argument('--time_dim',         type=int,   default=100,
                    help='Dimension of sinusoidal time encoding.')

parser.add_argument('--num_threads',      type=int,   default=8,
                    help='Total OMP threads for C++ sampler.')
parser.add_argument('--num_workers',      type=int,   default=1,
                    help='Number of sampler workers (num_thread_per_worker = num_threads // num_workers).')

parser.add_argument('--n_neighbor',       type=int,   default=10,
                    help='Neighbours sampled per layer.')
parser.add_argument('--strategy',         type=str, default='uniform',  
                    help="'recent' that samples most recent neighbors or 'uniform' that uniformly samples neighbors form the past")
parser.add_argument('--prop_time',        action='store_true', default=False, 
                    help="whether to use the timestamp of the root nodes when sampling for their multi-hop neighbors. Default stored as False")
parser.add_argument('--history',          type=int, default=1, 
                    help="number of snapshots to sample on")
parser.add_argument('--duration',         type=float, default=0.0, 
                    help="length in time of each snapshot, 0 for infinite length (used in non-snapshot-based methods")

parser.add_argument('--fold',             type=int,   default=0)
parser.add_argument('--prefix',           type=str,   default='')
parser.add_argument('--max_round',        type=int,   default=10)
parser.add_argument('--tolerance',        type=float, default=1e-4)
parser.add_argument('--pos_weight',       type=float, default=100,
                    help='Positive class weight. -1 = auto (n_neg/n_pos).')
parser.add_argument('--weight_decay',     type=float, default=5e-7)
parser.add_argument('--to_undirected',    action='store_true', default=False)
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
STRATEGY      = args.strategy
PROP_TIME     = args.prop_time
HISTORY       = args.history
DURATION      = args.duration

NUM_THREADS   = args.num_threads
NUM_WORKERS   = args.num_workers

MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
WEIGHT_DECAY  = args.weight_decay
EARLY_STOP_HIGHER_BETTER = args.early_stop_higher_better

MODEL_SAVE_PATH     = f'./saved_models/{args.prefix}-thegcn-with-sampler-node-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-thegcn-with-sampler-node-{DATA}-{epoch}.pt'

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh  = logging.FileHandler(os.path.join(DA.paths.log, f'{time.time()}.log'))
fh.setLevel(logging.DEBUG)
ch  = logging.StreamHandler()
ch.setLevel(logging.WARN)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt); ch.setFormatter(fmt)
logger.addHandler(fh); logger.addHandler(ch)
logger.info(args)

# ─────────────────────────────────────────────────────────────────────────────
# Load graph data (for node features, labels, masks, node_time)
# ─────────────────────────────────────────────────────────────────────────────

(graph, train_idx, val_idx, test_idx,
 train_labels_np, node_feat_dim) = load_dgraphfin_temporal(
    data_dir      = args.data_dir,
    fold          = args.fold,
    to_undirected = args.to_undirected,
)

# # Fix background node sentinels in node_time:
# # background nodes have large negative timestamps → replace with global max
# # so rel_t = node_time[dst] - edge_time is never a garbage large negative
# max_ts = graph.edge_time.float().max().item()
# node_time = graph.node_time.clone().float()
# node_time[node_time < 0] = max_ts
# graph.node_time = node_time

# Full node feature matrix and labels on CPU (indexed by global node id)
x_all         = graph.x                # [N, node_feat_dim]
y_all         = graph.y                # [N]
node_time_all = graph.node_time        # [N]  float

logger.info(
    f'Graph: {graph.num_nodes:,} nodes | {graph.num_edges:,} edges'
)
logger.info(
    f'Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,} | Background: {x_all.shape[0] - (len(train_idx) + len(val_idx) + len(test_idx)):,}'
)
logger.info(f'Train fraud rate: {train_labels_np.mean():.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# Load CSC sampler data and build TemporalSampler
# ─────────────────────────────────────────────────────────────────────────────

logger.info(f'Loading CSC sampler data from {args.sampler_dir}...')
g, df = load_graph(args.sampler_dir)

assert len(g['indptr']) == x_all.shape[0] + 1

# num_neighbors per layer: outer → inner, linearly decreasing
if NUM_LAYER == 0:
    num_neighbors = []      # TMP only
elif NUM_LAYER == 1:
    num_neighbors = [NUM_NEIGHBOR]
elif NUM_LAYER == 2:
    num_neighbors = [NUM_NEIGHBOR, 5]
else:
    step = max(1, (NUM_NEIGHBOR - 5) // (NUM_LAYER - 1))
    num_neighbors = [max(5, NUM_NEIGHBOR - i * step) for i in range(NUM_LAYER)]

sampler = ParallelSampler(
    g['indptr'], 
    g['indices'], 
    g['eid'], 
    g['ts'].astype(np.float32),
    NUM_THREADS, 
    NUM_WORKERS,                               # num_workers
    NUM_LAYER, 
    num_neighbors,
    STRATEGY=='recent', 
    PROP_TIME,
    HISTORY, 
    float(DURATION)
)
logger.info(f'Sampler: {sampler}')
logger.info(f'NeighborLoader fanouts (outer→inner): {num_neighbors}')

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
x_all         = x_all.to(device)
node_time_all = node_time_all.to(device)
y_all         = y_all.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = THEGCNSamplerModel(
    in_channels     = node_feat_dim,
    hidden_channels = NODE_DIM,
    n_smp_layers    = NUM_LAYER,
    time_dim        = TIME_DIM,
    dropout         = DROP_OUT,
).to(device)

logger.info(
    f'THEGCN | smp_layers: {NUM_LAYER} | hidden: {NODE_DIM} | '
    f'time_dim: {TIME_DIM} | '
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

# # ─────────────────────────────────────────────────────────────────────────────
# # Helper: build mini-batch tensors from a SampledBlock
# # ─────────────────────────────────────────────────────────────────────────────

# def block_to_tensors(blocks):
#     """
#     Convert list of SampledBlock (one per sampler layer) into tensors for
#     THEGCNModel.forward().

#     The innermost block (blocks[-1]) contains the direct neighbours of seeds
#     and is used for the TMP pass. The outer blocks (blocks[:-1]) are passed
#     to the SMP layers in order outermost → innermost.

#     Returns
#     -------
#     x          : [N_sub, feat_dim]   node features for subgraph
#     edge_index : [2, E]              local edge index (innermost block)
#     dts        : [E]                 Δt = query_ts[dst] - edge_ts  (for TimeEncode)
#     n_seeds    : int                 number of seed nodes
#     """
#     # Use innermost block for TMP
#     inner = blocks[-1]
#     global_ids = inner.nodes.to(device)        # [N_sub]  global node ids
#     x          = x_all[global_ids]             # [N_sub, feat]
#     edge_index = inner.edge_index.to(device)   # [2, E]
#     dts        = inner.dts.to(device)          # [E]  precomputed Δt
#     n_seeds    = inner.n_seeds

#     return x, edge_index, dts, n_seeds, global_ids


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_nodes(split_idx):
    model.eval()
    all_preds, all_labels = [], []

    # iterate in batches
    idx_array = split_idx.cpu().numpy()
    for start in range(0, len(idx_array), BATCH_SIZE * 2):
        batch_idx = idx_array[start : start + BATCH_SIZE * 2]
        root_nodes = batch_idx.astype(np.int32)
        ts         = node_time_all[batch_idx].cpu().numpy().astype(np.float32)
        
        sampler.sample(root_nodes, ts)
        ret        = sampler.get_ret()
        mfgs       = to_dgl_blocks(ret, hist=HISTORY, reverse=False, cuda=(device.type != 'cpu'))
        batch_inputs = to_pyg_inputs(mfgs, device=device)

        n_seeds = batch_inputs[-1][0]['size'][1]   # num_dst_nodes of innermost block
        labels  = y_all[batch_inputs[-1][0]['node_ids'][:n_seeds]]
 
        logits = model(
            x_all        = x_all,
            batch_inputs = batch_inputs,
            batch_size   = n_seeds,
        )

        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

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
    # rc1 = recall_at_top_n_percent(all_labels, scores, 1)
    # rc10 = recall_at_top_n_percent(all_labels, scores, 10)

    val_loss = torch.nn.BCEWithLogitsLoss()(
        torch.tensor(all_preds), torch.tensor(all_labels)
    ).item()

    return auc, ap, f1, mcc, rc, pr, val_loss

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

early_stopper    = EarlyStopMonitor(
    max_round    = MAX_ROUND,
    higher_better = EARLY_STOP_HIGHER_BETTER,
    tolerance    = TOLERANCE,
)
last_saved_epoch = -1
train_loss_hist  = []
val_loss_hist    = []
val_auc_hist     = []

train_idx_np = train_idx.cpu().numpy()

mlflow.set_experiment('thegcn-sampler')
with mlflow.start_run():
    mlflow.log_params(vars(args))
    
    for epoch in range(NUM_EPOCH):
        logger.info(f"Epoch {epoch+1}:")
        model.train()

        time_sample = 0
        time_prep = 0
        m_loss = []

        if sampler is not None:
            sampler.reset()
        
        # shuffle training indices each epoch
        perm     = np.random.permutation(len(train_idx_np))
        idx_perm = train_idx_np[perm]

        pbar = tqdm(range(0, len(idx_perm), BATCH_SIZE), desc=f'Epoch {epoch}')
        for start in pbar:
            batch_idx  = idx_perm[start : start + BATCH_SIZE]
            root_nodes = batch_idx.astype(np.int32)
            ts         = node_time_all[batch_idx].cpu().numpy().astype(np.float32)
            # sampling
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            time_sample += ret[0].sample_time()
            
            # message flow graphs from sampled blocks
            mfgs = to_dgl_blocks(ret, hist=HISTORY, reverse=False, cuda=(device != 'cpu'))
            # to pyg inputs
            batch_inputs = to_pyg_inputs(mfgs, device=device)

            n_seeds = batch_inputs[-1][0]['size'][1]
            labels  = y_all[batch_inputs[-1][0]['node_ids'][:n_seeds]]


            logits = model(
                x_all        = x_all,
                batch_inputs = batch_inputs,
                batch_size   = n_seeds,
            )
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr, val_loss = eval_nodes(val_idx)
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

        if early_stopper.early_stop_check(val_loss):
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
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, _ = eval_nodes(test_idx)
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
    fig, ax1    = plt.subplots(figsize=(10, 5))

    color_train = '#e05c5c'
    color_val   = '#e09c5c'
    color_auc   = '#5c8de0'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_train)
    ax1.plot(epochs_list, train_loss_hist, color=color_train, linewidth=2, label='Train Loss')
    ax1.plot(epochs_list, val_loss_hist,   color=color_val,   linewidth=2, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_train)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val AUC', color=color_auc)
    ax2.plot(epochs_list, val_auc_hist, color=color_auc, linewidth=2, linestyle='--', label='Val AUC')
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.set_ylim(0, 1)

    best_ep = early_stopper.best_epoch
    ax2.axvline(x=best_ep, color='gray', linestyle=':', linewidth=1.5,
                label=f'Best epoch ({best_ep})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('THEGCN (TGL sampler) — Training Curve')
    plt.tight_layout()

    plot_path = f'./saved_models/{args.prefix}-thegcn-sampler-node-{DATA}-training-curve.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    mlflow.log_artifact(plot_path)
    logger.info(f'Training curve saved to {plot_path}')