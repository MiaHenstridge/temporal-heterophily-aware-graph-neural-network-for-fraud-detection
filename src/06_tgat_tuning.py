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

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import TransformerConv

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import EarlyStopMonitor
from namespaces import DA
from dgraphfin import DGraphFin   # reuse existing dataset class

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
parser.add_argument('--n_neighbor',      type=int,   default=20,
                    help='neighbors sampled per layer in NeighborLoader')
parser.add_argument('--fold',            type=int,   default=0)
parser.add_argument('--prefix',          type=str,   default='')
parser.add_argument('--max_round',       type=int,   default=10)
parser.add_argument('--tolerance',       type=float, default=1e-4)
parser.add_argument('--pos_weight',      type=float, default=100,
                    help='positive class weight; -1 = auto (n_neg/n_pos)')
parser.add_argument('--num_workers',     type=int,   default=12)
parser.add_argument('--weight_decay',    type=float, default=5e-7)
parser.add_argument('--max_time_steps',  type=int,   default=32,
                    help='discretisation scale for edge timestamps (default 32)')

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
N_HEAD        = args.n_head
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
NUM_WORKERS   = args.num_workers
WEIGHT_DECAY  = args.weight_decay

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
# Temporal data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dgraphfin_temporal(data_dir: str, fold: int = 0, max_time_steps: int = 32):
    """
    Extend the existing DGraphFin dataset with temporal information for TGAT,
    following the original process_data() logic exactly.

    Edge-time processing (process_data steps 1-4)
    ----------------------------------------------
    1. Shift:       edge_time -= edge_time.min()          (start from 0)
    2. Normalise:   edge_time /= edge_time.max()          (range [0, 1])
    3. Discretise:  edge_time = (edge_time * max_time_steps).long()
    4. Reshape:     edge_time = edge_time.view(-1, 1).float()   → [E, 1]

    Node-time (process_data groupby-min)
    -------------------------------------
    node_time[v] = min(edge_time) over all out-edges of v.
    Nodes with no out-edges get 0.

    Symmetrisation (process_data)
    ------------------------------
    Reverse edges are added as edge_index[[1,0],:].
    The SAME edge_time is reused for both directions (no Δt computation).

    Parameters
    ----------
    data_dir : str
    fold : int
    max_time_steps : int   Discretisation scale; default 32 matches the original.
    """
    import pandas as pd
    import torch_geometric.transforms as T

    # ── load cached dataset (shared with static baselines) ──────────────────
    dataset = DGraphFin(root=data_dir, pre_transform=T.ToSparseTensor())
    data    = dataset[0]

    # ── z-score normalise features ───────────────────────────────────────────
    x      = data.x
    x      = (x - x.mean(0)) / x.std(0)
    data.x = x

    # ── squeeze label tensor ─────────────────────────────────────────────────
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    node_feat_dim = data.num_features
    N             = data.num_nodes

    # ── recover directed edge_index from adj_t ───────────────────────────────
    # adj_t convention: row=dst, col=src  →  src=col, dst=row
    row_t, col_t, _ = data.adj_t.coo()
    src = col_t                                             # [E]
    dst = row_t                                             # [E]
    edge_index_directed = torch.stack([src, dst], dim=0)   # [2, E]

    # ── Steps 1-4: edge_time processing (exactly as process_data) ────────────
    et = data.edge_time.float()
    et = et - et.min()                       # 1. shift
    et = et / et.max()                       # 2. normalise
    et = (et * max_time_steps).long()        # 3. discretise
    et = et.view(-1, 1).float()              # 4. reshape  → [E, 1]

    # ── node_time via groupby-min (exactly as process_data) ─────────────────
    # Original stacks [edge_index (2×E), edge_time (1×E)] → [3, E] then
    # transposes to DataFrame and does groupby(src_col).min().
    edge_np  = torch.cat(
        [edge_index_directed, et.view(1, -1)], dim=0
    ).T.numpy()                              # [E, 3]: src | dst | time

    df          = pd.DataFrame(edge_np, columns=['src', 'dst', 'time'])
    min_time_df = df.groupby('src')['time'].min()   # Series: src_id → min_time

    node_time_np = np.zeros(N, dtype=np.float32)
    node_time_np[min_time_df.index.astype(int)] = min_time_df.values.astype(np.float32)
    node_time = torch.tensor(node_time_np)   # [N]

    # ── symmetrise (exactly as process_data) ─────────────────────────────────
    # edge_index[[1,0],:] flips src↔dst; edge_time is simply repeated.
    edge_index_sym = torch.cat(
        [edge_index_directed, edge_index_directed[[1, 0], :]], dim=1
    )                                        # [2, 2E]
    edge_attr_sym  = torch.cat([et, et], dim=0)   # [2E, 1]

    # ── split masks ──────────────────────────────────────────────────────────
    train_idx = data.train_mask
    val_idx   = data.valid_mask
    test_idx  = data.test_mask

    if train_idx.dim() > 1 and train_idx.shape[1] > 1:
        train_idx = train_idx[:, fold]
        val_idx   = val_idx[:, fold]
        test_idx  = test_idx[:, fold]

    # ── binary labels ────────────────────────────────────────────────────────
    y_binary     = (data.y == 1).float()
    train_labels = y_binary[train_idx].numpy()

    # ── assemble graph ───────────────────────────────────────────────────────
    graph = Data(
        x          = data.x,             # [N, 17]   z-score normalised
        edge_index = edge_index_sym,     # [2, 2E]   bidirectional
        edge_attr  = edge_attr_sym,      # [2E, 1]   discretised edge time
        node_time  = node_time,          # [N]       min out-edge time per node
        y          = y_binary,           # [N]       binary fraud label
        num_nodes  = N,
    )

    return graph, train_idx, val_idx, test_idx, train_labels, node_feat_dim


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time encoding  (Bochner / TGAT paper eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

class TimeEncode(torch.nn.Module):
    """
    Learnable sinusoidal time encoding from the original TGAT implementation:
    https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

    Maps a scalar time value t to a time_dim-dimensional vector:

        φ(t) = cos(t · basis_freq + phase)

    where basis_freq and phase are **learned** parameters initialised as:
        basis_freq  ~ log-spaced  1 / 10^linspace(0, 9, time_dim)
        phase       ~ zeros

    Parameters
    ----------
    expand_dim : int
        Output dimensionality (called time_dim elsewhere in this file).
    factor : int
        Stored for compatibility with the original API; not used in forward.
    """

    def __init__(self, expand_dim: int, factor: int = 5):
        super().__init__()
        time_dim    = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim)
            ).float()
        )                                             # [time_dim]  learned
        self.phase = torch.nn.Parameter(
            torch.zeros(time_dim).float()
        )                                             # [time_dim]  learned

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts : Tensor  shape [N, L]
            Raw time scalars arranged as a 2-D batch
            (N = nodes/edges, L = sequence length; L=1 for single timestamps).

        Returns
        -------
        Tensor  shape [N, L, time_dim]
        """
        batch_size = ts.size(0)
        seq_len    = ts.size(1)

        ts      = ts.view(batch_size, seq_len, 1)              # [N, L, 1]
        map_ts  = ts * self.basis_freq.view(1, 1, -1)          # [N, L, time_dim]
        map_ts  = map_ts + self.phase.view(1, 1, -1)           # [N, L, time_dim]

        harmonic = torch.cos(map_ts)                           # [N, L, time_dim]
        return harmonic


# ─────────────────────────────────────────────────────────────────────────────
# TGAT model
# ─────────────────────────────────────────────────────────────────────────────

class TGATModel(torch.nn.Module):
    """
    Temporal Graph Attention Network for node classification.

    Each layer:
        1. Encode the scalar Δt on each edge → edge embedding ∈ R^time_dim
        2. Concatenate node feature with its node_time encoding → augmented input
        3. TransformerConv with edge_dim=time_dim (multi-head dot-product attn)
        4. BN → ReLU → Dropout
    MLP classifier head identical to the static GNN baselines.

    Parameters
    ----------
    in_channels : int
        Raw node feature dimension (17 for DGraphFin).
    hidden_channels : int
        Hidden dimension after each conv layer.
    n_layers : int
        Number of TGAT layers to stack.
    n_head : int
        Number of attention heads (hidden_channels must be divisible by n_head).
    time_dim : int
        Dimension of sinusoidal time encoding used for both node and edge times.
    dropout : float
        Dropout probability applied after each conv layer and in the MLP.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_layers:        int,
        n_head:          int,
        time_dim:        int,
        dropout:         float = 0.2,
    ):
        super().__init__()
        assert hidden_channels % n_head == 0, \
            f'hidden_channels ({hidden_channels}) must be divisible by n_head ({n_head})'

        self.time_enc = TimeEncode(time_dim)
        self.dropout  = dropout
        self.n_layers = n_layers

        # Each conv expects:  node features  +  time encoding of the node
        # → input dim = in_channels + time_dim for layer 0,
        #              hidden_channels + time_dim for subsequent layers.
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        first_in = in_channels + time_dim
        self.convs.append(
            TransformerConv(
                in_channels  = first_in,
                out_channels = hidden_channels // n_head,
                heads        = n_head,
                edge_dim     = time_dim,   # edge_attr will be time-encoded Δt
                dropout      = dropout,
                concat       = True,
                beta         = False,
            )
        )
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 1):
            self.convs.append(
                TransformerConv(
                    in_channels  = hidden_channels + time_dim,
                    out_channels = hidden_channels // n_head,
                    heads        = n_head,
                    edge_dim     = time_dim,
                    dropout      = dropout,
                    concat       = True,
                    beta         = False,
                )
            )
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # MLP classifier head — identical structure to static GNN baselines
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x:          torch.Tensor,   # [N_sub, in_channels]
        edge_index: torch.Tensor,   # [2, E_sub]
        edge_attr:  torch.Tensor,   # [E_sub, 1]   raw Δt scalars
        node_time:  torch.Tensor,   # [N_sub]       node timestamps
        batch_size: int,
    ) -> torch.Tensor:              # [batch_size]  logits

        # ── encode edge time deltas ───────────────────────────────────────────
        # TimeEncode expects [N, L]; edge_attr is [E, 1] so L=1.
        # Output: [E, 1, time_dim] → squeeze to [E, time_dim]
        edge_time_enc = self.time_enc(edge_attr).squeeze(1)        # [E, time_dim]

        # ── encode node timestamps ────────────────────────────────────────────
        # node_time: [N] → unsqueeze → [N, 1] → TimeEncode → [N, 1, time_dim]
        # → squeeze → [N, time_dim]
        node_time_enc = self.time_enc(
            node_time.unsqueeze(1)
        ).squeeze(1)                                               # [N, time_dim]

        # ── first layer ───────────────────────────────────────────────────────
        h = torch.cat([x, node_time_enc], dim=-1)                  # [N, in + time_dim]
        h = self.norms[0](self.convs[0](h, edge_index, edge_attr=edge_time_enc).relu())
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ── subsequent layers ─────────────────────────────────────────────────
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            h = torch.cat([h, node_time_enc], dim=-1)              # [N, hidden + time_dim]
            h = norm(conv(h, edge_index, edge_attr=edge_time_enc).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)

        # ── classify seed nodes only ──────────────────────────────────────────
        return self.clf(h[:batch_size]).squeeze(-1)         # [batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

(graph, train_idx, val_idx, test_idx,
 train_labels_np, node_feat_dim) = load_dgraphfin_temporal(
    data_dir=args.data_dir, fold=args.fold, max_time_steps=args.max_time_steps
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
    num_neighbors = [NUM_NEIGHBOR, 10]
else:
    step = max(1, (NUM_NEIGHBOR - 10) // (NUM_LAYER - 1))
    num_neighbors = [max(10, NUM_NEIGHBOR - i * step) for i in range(NUM_LAYER)]

logger.info(f'NeighborLoader fanouts (outer→inner): {num_neighbors}')

# NeighborLoader propagates all node/edge attributes automatically because
# graph.edge_attr and graph.node_time are registered as Data attributes.
train_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    batch_size    = BATCH_SIZE,
    input_nodes   = train_idx,
    shuffle       = True,
    num_workers   = NUM_WORKERS,
)

val_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    batch_size    = BATCH_SIZE * 2,
    input_nodes   = val_idx,
    shuffle       = False,
    num_workers   = NUM_WORKERS,
)

test_loader = NeighborLoader(
    graph,
    num_neighbors = num_neighbors,
    batch_size    = BATCH_SIZE * 2,
    input_nodes   = test_idx,
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
            edge_attr  = batch.edge_attr,
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

early_stopper    = EarlyStopMonitor(max_round=MAX_ROUND, higher_better=False, tolerance=TOLERANCE)
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
                edge_attr  = batch.edge_attr,
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