"""
06_tgat_tuning.py
===================
TGAT training script using the TGL C++ parallel sampler for temporally-
correct neighbourhood sampling.

Prerequisites
-------------
1. Build the C++ sampler extension:
       python src/setup.py build_ext --inplace

2. Prepare the CSC graph data (one-time):
       python src/tgl_data_preprocess.py

3. Run training:
       python src/06_tgat_tuning_with_sampler.py --prefix run1 --n_epoch 50 --bs 1024 \\
           --n_layer 2 --node_dim 128 --time_dim 100 --n_neighbor 10 --gpu 0
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

from models import TGATModel
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

parser = argparse.ArgumentParser('TGAT Node Classification on DGraphFin (TGL sampler)')
parser.add_argument('-d', '--data',       type=str,   default='DGraphFin')
parser.add_argument('--data_dir',         type=str,   default='./datasets')
parser.add_argument('--sampler_dir',     type=str,   default='./processed_data/tgl',
                    help='Path to CSC graph .npz built by tgl_data_preprocess.py')
parser.add_argument('--bs',               type=int,   default=512)
parser.add_argument('--n_epoch',          type=int,   default=100)
parser.add_argument('--lr',               type=float, default=1e-3)
parser.add_argument('--drop_out',         type=float, default=0.2)
parser.add_argument('--gpu',              type=int,   default=0)
parser.add_argument('--n_layer',          type=int,   default=2,
                    help='Number of GATConv latyers.')
parser.add_argument('--node_dim',         type=int,   default=128,
                    help='Hidden dimension for GATConv layers.')
parser.add_argument('--time_dim',         type=int,   default=100,
                    help='Dimension of sinusoidal time encoding.')
parser.add_argument('--feat_augment',     action='store_true', default=False,
                    help='whether to augment features')
parser.add_argument('--n_head',           type=int,   default=4,)

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

parser.add_argument('--loss',            type=str, default='bce',
                    help='loss function to use for training (bce or focal)')
parser.add_argument('--pos_weight',      type=float, default=100,
                    help='positive class weight; -1 = auto (n_neg/n_pos)')
parser.add_argument('--alpha',           type=float, default=0.95,
                    help="weight for the positive (fraud) class (only use when loss='focal'). Must be between (0, 1)")
parser.add_argument('--gamma',           type=float, default=2.0,
                    help="focusing exponent (only use when loss='focal'). Must be >= 0")
parser.add_argument('--reduction',       type=str, default='mean',
                    help="'mean' (default), 'sum', or 'none' (only use when loss='focal')")

parser.add_argument('--weight_decay',     type=float, default=5e-7)
parser.add_argument('--to_undirected',    action='store_true', default=False)
parser.add_argument('--early_stop_higher_better', action='store_true', default=False)
parser.add_argument('--seed',             type=int, default=1111)

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit(0)

RANDOM_SEED = args.seed
set_seed(RANDOM_SEED)

BATCH_SIZE    = args.bs
NUM_EPOCH     = args.n_epoch
LEARNING_RATE = args.lr
DROP_OUT      = args.drop_out
GPU           = args.gpu
DATA          = args.data

NODE_DIM      = args.node_dim
TIME_DIM      = args.time_dim
FEAT_AUGMENT  = args.feat_augment
NUM_LAYER     = args.n_layer
NUM_HEAD      = args.n_head
NUM_NEIGHBOR  = args.n_neighbor
STRATEGY      = args.strategy
PROP_TIME     = args.prop_time
HISTORY       = args.history
DURATION      = args.duration

NUM_THREADS   = args.num_threads
NUM_WORKERS   = args.num_workers

MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
LOSS          = args.loss
WEIGHT_DECAY  = args.weight_decay
EARLY_STOP_HIGHER_BETTER = args.early_stop_higher_better

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
# Load CSC sampler data for sampler (separate for train/val/test)
# ─────────────────────────────────────────────────────────────────────────────

logger.info(f'Loading CSC sampler data from {args.sampler_dir} for train/val/tests...')
# build all three samplers once before training
g_train, _ = load_graph(args.sampler_dir, mode='train')
g_val, _   = load_graph(args.sampler_dir, mode='val')
g_test, _  = load_graph(args.sampler_dir, mode='test')

# num_neighbors per layer: outer → inner, linearly decreasing
if NUM_LAYER == 1:
    num_neighbors = [NUM_NEIGHBOR]
else:
    num_neighbors = [NUM_NEIGHBOR] * NUM_LAYER

sampler_train = ParallelSampler(
    g_train['indptr'], 
    g_train['indices'], 
    g_train['eid'], 
    g_train['ts'].astype(np.float32),
    NUM_THREADS, 
    NUM_WORKERS,                               # num_workers
    NUM_LAYER, 
    num_neighbors,
    STRATEGY=='recent', 
    PROP_TIME,
    HISTORY, 
    float(DURATION)
)

sampler_val   = ParallelSampler(
    g_val['indptr'], 
    g_val['indices'], 
    g_val['eid'], 
    g_val['ts'].astype(np.float32),
    NUM_THREADS, 
    NUM_WORKERS,                               # num_workers
    NUM_LAYER, 
    num_neighbors,
    STRATEGY=='recent', 
    PROP_TIME,
    HISTORY, 
    float(DURATION)
)

sampler_test  = ParallelSampler(
    g_test['indptr'], 
    g_test['indices'], 
    g_test['eid'], 
    g_test['ts'].astype(np.float32),
    NUM_THREADS, 
    NUM_WORKERS,                               # num_workers
    NUM_LAYER, 
    num_neighbors,
    STRATEGY=='recent', 
    PROP_TIME,
    HISTORY, 
    float(DURATION)
)


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

model = TGATModel(
    in_channels     = node_feat_dim,
    hidden_channels = NODE_DIM,
    n_layers        = NUM_LAYER,
    n_head          = NUM_HEAD,
    time_dim        = TIME_DIM,
    dropout         = DROP_OUT,
    feature_augment = FEAT_AUGMENT,
).to(device)

logger.info(
    f'TGAT | num_layers: {NUM_LAYER} | hidden: {NODE_DIM} | '
    f'time_dim: {TIME_DIM} | '
    f'params: {sum(p.numel() for p in model.parameters()):,}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Loss / optimiser
# ─────────────────────────────────────────────────────────────────────────────

if LOSS == 'bce':
    n_neg = (train_labels_np == 0).sum()
    n_pos = (train_labels_np == 1).sum()
    pw    = float(n_neg) / float(n_pos) if args.pos_weight < 0 else args.pos_weight
    logger.info(f'Using BCEWithLogitsLoss with pos_weight: {pw:.2f}')
    pos_weight = torch.tensor([pw], dtype=torch.float, device=device)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
elif LOSS == 'focal':
    logger.info(f"Using BalancedFocalLoss with alpha: {args.alpha:.2f}, gamma: {args.gamma:.2f}, reduction: {args.reduction}")
    criterion = BalancedFocalLoss(alpha=args.alpha, gamma=args.gamma, reduction=args.reduction).to(device)


optimizer  = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-6
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_nodes(split_idx, criterion, sampler):
    model.eval()

    if sampler is not None:
        sampler.reset()
        
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
    rc_at_127  = recall_at_top_n_percent(all_labels, scores, 1.27)
    pr_at_127  = precision_at_top_n_percent(all_labels, scores, 1.27)

    val_loss = criterion(
        torch.tensor(all_preds, device=device),
        torch.tensor(all_labels, device=device),
    ).item()

    return auc, ap, f1, mcc, rc, pr, rc_at_127, pr_at_127, val_loss

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

# create experiment
EXPERIMENT_NAME = 'temporal-gnn-tgat-v2'
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_SAVE_PATH     = f'./saved_models/{EXPERIMENT_NAME}-{args.prefix}-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{EXPERIMENT_NAME}-{args.prefix}-{DATA}.pt'

with mlflow.start_run():
    mlflow.log_params(vars(args))
    mlflow.set_tag('seed', RANDOM_SEED)
    
    for epoch in range(NUM_EPOCH):
        logger.info(f"Epoch {epoch+1}:")
        model.train()

        time_sample = 0
        time_prep = 0
        m_loss = []

        if sampler_train is not None:
            sampler_train.reset()
        
        # shuffle training indices each epoch
        perm     = np.random.permutation(len(train_idx_np))
        idx_perm = train_idx_np[perm]

        pbar = tqdm(range(0, len(idx_perm), BATCH_SIZE), desc=f'Epoch {epoch}')
        for start in pbar:
            batch_idx  = idx_perm[start : start + BATCH_SIZE]
            root_nodes = batch_idx.astype(np.int32)
            ts         = node_time_all[batch_idx].cpu().numpy().astype(np.float32)
            # sampling
            sampler_train.sample(root_nodes, ts)
            ret = sampler_train.get_ret()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        # validation
        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr, _, _, val_loss = eval_nodes(val_idx, criterion, sampler_val)
        scheduler.step(val_ap)


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

        if early_stopper.early_stop_check(val_ap):
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
                # ── track best val metrics ────────────────────────────────
                best_val_metrics = {
                    'best_epoch':    epoch,
                    'best_val_loss':      val_loss,
                    'best_val_auc':       val_auc,
                    'best_val_ap':        val_ap,
                    'best_val_f1':        val_f1,
                    'best_val_mcc':       val_mcc,
                    'best_val_recall':    val_rc,
                    'best_val_precision': val_pr,
                }
        
    # ── log best val metrics and final test after training ends ──────────
    mlflow.log_metrics(best_val_metrics)

    # ── final test ───────────────────────────────────────────────────────────
    # Reload the best checkpoint into the model before testing and final saving
    logger.info(f"Loading best model from epoch {early_stopper.best_epoch} for testing.")
    model.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))

    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, test_rc_at_127, test_pr_at_127, _ = eval_nodes(test_idx, criterion, sampler_test)
    logger.info(
        f'Test | auc: {test_auc:.4f} | ap: {test_ap:.4f} | '
        f'f1: {test_f1:.4f} | mcc: {test_mcc:.4f} | '
        f'recall: {test_rc:.4f} | precision: {test_pr:.4f} | recall @ top 1.27%: {test_rc_at_127:.4f} | precision @ top 1.27%: {test_rc_at_127:.4f}'
    )
    mlflow.log_metrics({
        'test_auc':       test_auc,
        'test_ap':        test_ap,
        'test_f1':        test_f1,
        'test_mcc':       test_mcc,
        'test_recall':    test_rc,
        'test_precision': test_pr,
        'test_recall_top127': test_rc_at_127,
        'test_precision_top127': test_pr_at_127,
    })

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')