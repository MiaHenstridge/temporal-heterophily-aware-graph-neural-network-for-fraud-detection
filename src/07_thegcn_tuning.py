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

from models import THEGCNModel

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import EarlyStopMonitor
from namespaces import DA
from dgraphfin import DGraphFin, load_dgraphfin_temporal  # reuse existing dataset class
from sampler import TemporalSampler, load_sampler_data

# ── working directory: project root ──────────────────────────────────────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log,         exist_ok=True)
os.makedirs('./saved_models',     exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser('THEGCN Node Classification on DGraphFin')
parser.add_argument('-d', '--data',      type=str,   default='DGraphFin')
parser.add_argument('--data_dir',        type=str,   default='./datasets')
parser.add_argument('--bs',              type=int,   default=1024)
parser.add_argument('--n_epoch',         type=int,   default=200)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--drop_out',        type=float, default=0.2)
parser.add_argument('--gpu',             type=int,   default=0)
parser.add_argument('--n_layer',         type=int,   default=2)     # number of SMP layers
parser.add_argument('--node_dim',        type=int,   default=128,
                    help='hidden dimension (must be divisible by n_head)')
parser.add_argument('--time_dim',        type=int,   default=100,
                    help='dimension of the sinusoidal time encoding')
# parser.add_argument('--n_head',          type=int,   default=2,
#                     help='number of attention heads in TransformerConv')
parser.add_argument('--n_neighbor',      type=int,   default=10,
                    help='neighbors sampled per layer in ParallelSampler')

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
# TEMPORAL_STRATEGY = args.temporal_strategy
# N_HEAD        = args.n_head
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
NUM_WORKERS   = args.num_workers
WEIGHT_DECAY  = args.weight_decay
EARLY_STOP_HIGHER_BETTER = args.early_stop_higher_better


MODEL_SAVE_PATH     = f'./saved_models/thegcn-{args.prefix}-node-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/thegcn-{args.prefix}-node-{DATA}-{epoch}.pt'

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
# Load node features and splits
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading node features and masks...")
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
# Load data for sampler
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading preprocessed data for parallel sampling...")
sampler_input = load_sampler_data(os.path.join(DA.paths.output_data_tgl, 'dgraphfin_tgl.npz'))


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Event Sampler
# ─────────────────────────────────────────────────────────────────────────────
if NUM_LAYER == 1:
    num_neighbors = [NUM_NEIGHBOR]
elif NUM_LAYER == 2:
    num_neighbors = [NUM_NEIGHBOR, 5]
else:
    step = max(1, (NUM_NEIGHBOR - 5) // (NUM_LAYER - 1))
    num_neighbors = [max(5, NUM_NEIGHBOR - i * step) for i in range(NUM_LAYER)]

logger.info(f'NeighborLoader fanouts (outer→inner): {num_neighbors}')

sampler = TemporalSampler(
    graph_data=sampler_input,
    num_neighbors=num_neighbors,
    num_workers=8,
    num_threads=8,
    recent=True,
)

