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
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
from utils import EarlyStopMonitor
from namespaces import DA

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log, exist_ok=True)
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

################## argument parser ##################
parser = argparse.ArgumentParser('TGN Direct Node Classification')
parser.add_argument('-d', '--data',      type=str,   default='dgraphfin')
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
parser.add_argument('--max_round',       type=int,   default=5)
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
TIME_DIM      = args.time_dim
MEMORY_DIM    = args.memory_dim
NUM_NEIGHBOR  = args.n_neighbor
NUM_LAYER     = args.n_layer
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
POS_WEIGHT    = args.pos_weight

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

# fast label lookup for temporal training: node_idx -> label, -1 = no label
train_label_arr = np.full(max_idx + 2, -1.0, dtype=np.float32)

# populate label lookup
train_label_arr[train_nodes] = train_labels

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

train_edge_flag = train_arr[src_l] & train_arr[dst_l]   # both src and dst in train set (no leakage)
val_edge_flag   = val_arr[src_l]   & val_arr[dst_l]     # both src and dst in val set (no leakage)
test_edge_flag  = test_arr[src_l]  & test_arr[dst_l]    # both src and dst in test set (no leakage)

nn_val_edge_flag = (train_arr[src_l] & val_arr[dst_l]) | (val_arr[src_l] & train_arr[dst_l])  # edges connecting train and val nodes (for val replay)
nn_test_edge_flag = (train_arr[src_l] & test_arr[dst_l]) | (test_arr[src_l] & train_arr[dst_l]) | (val_arr[src_l] & test_arr[dst_l]) | (test_arr[src_l] & val_arr[dst_l])  # edges connecting test nodes to train/val nodes (for test replay)

# temporal train loader — all edges in time order, labelled nodes get supervised
train_edge_data   = full_data[train_edge_flag]
train_edge_loader = TemporalDataLoader(train_edge_data, batch_size=BATCH_SIZE)

logger.info(f'Total edges: {len(src_l)}')
logger.info(f'Train edges: {train_edge_flag.sum()}, Val edges: {val_edge_flag.sum() + nn_val_edge_flag.sum()}, Test edges: {test_edge_flag.sum() + nn_test_edge_flag.sum()}')

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

# node classifier head — takes GNN embedding + raw node features as skip connection
clf_input_dim = NODE_DIM + node_feat_dim
node_clf = torch.nn.Sequential(
    torch.nn.Linear(clf_input_dim, NODE_DIM),
    torch.nn.BatchNorm1d(NODE_DIM),
    torch.nn.ReLU(),
    torch.nn.Dropout(DROP_OUT),
    torch.nn.Linear(NODE_DIM, NODE_DIM // 2),
    torch.nn.BatchNorm1d(NODE_DIM // 2),
    torch.nn.ReLU(),
    torch.nn.Dropout(DROP_OUT),
    torch.nn.Linear(NODE_DIM // 2, 1),
).to(device)

neighbor_loader = LastNeighborLoader(max_idx + 2, size=NUM_NEIGHBOR, device=device)

# node feature projection: 17-dim -> MEMORY_DIM
node_feats_th  = torch.tensor(node_feats, dtype=torch.float).to(device)
node_feat_proj = torch.nn.Linear(node_feat_dim, MEMORY_DIM).to(device)

# pos_weight for class imbalance
n_neg = (train_labels == 0).sum()
n_pos = (train_labels == 1).sum()
if POS_WEIGHT > 0:
    pw = POS_WEIGHT
else:
    pw = n_neg / n_pos   # auto
    logger.info(f'Auto pos_weight: {pw:.2f} (n_neg={n_neg}, n_pos={n_pos})')
pos_weight = torch.tensor([pw], dtype=torch.float).to(device)
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
    Returns concatenation of GNN embedding and raw node features.
    nodes_tensor: 1D LongTensor of global node ids (1-based).
    Returns: (N, NODE_DIM + node_feat_dim) tensor.
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
            safe_e_id = e_id_valid % len(ts_l)
            edge_t   = ts_l[safe_e_id].to(device)
            edge_msg = msg[safe_e_id].to(device)
            z = gnn(z, last_update, local_edge_index, edge_t, edge_msg)

    z_nodes = z[assoc[nodes_tensor]]

    # skip connection: concatenate raw node features
    raw_feats = node_feats_th[nodes_tensor]
    return torch.cat([z_nodes, raw_feats], dim=-1)


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
    all_preds, all_labels, all_losses = [], [], []

    for s in range(0, num_instance, BATCH_SIZE):
        e = min(num_instance, s + BATCH_SIZE)
        node_batch   = torch.tensor(nodes[s:e],    dtype=torch.long,  device=device)
        label_batch  = labels_np[s:e]
        label_tensor = torch.tensor(label_batch,   dtype=torch.float, device=device)

        z    = get_node_embeddings(node_batch)
        pred = node_clf(z).squeeze(-1)

        all_losses.append(criterion(pred, label_tensor).item())
        all_preds.append(pred.cpu().numpy())
        all_labels.append(label_batch)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    scores     = torch.sigmoid(torch.tensor(all_preds)).numpy()
    pred_label = (scores > 0.5).astype(int)

    auc      = roc_auc_score(all_labels, scores)
    ap       = average_precision_score(all_labels, scores)
    f1       = f1_score(all_labels, pred_label, zero_division=0)
    mcc      = matthews_corrcoef(all_labels, pred_label)
    rc       = recall_score(all_labels, pred_label, zero_division=0)
    pr       = precision_score(all_labels, pred_label, zero_division=0)
    val_loss = np.mean(all_losses)

    return auc, ap, f1, mcc, rc, pr, val_loss


################## training ##################
early_stopper = EarlyStopMonitor(max_round=MAX_ROUND, higher_better=True, tolerance=TOLERANCE)
last_saved_epoch = -1  # track which epoch was last checkpointed

# history for plotting
train_loss_hist = []
val_loss_hist   = []
val_auc_hist    = []

mlflow.set_experiment('tgn-node-direct')
with mlflow.start_run():
    mlflow.log_params(vars(args))

    # precompute val replay flag — constant across epochs
    val_replay_flag = val_edge_flag | train_edge_flag | nn_val_edge_flag # replay all train+val edges for val evaluation (no leakage)

    for epoch in range(NUM_EPOCH):
        tgn_memory.train()
        gnn.train()
        node_clf.train()

        # reset memory at start of each epoch
        tgn_memory.reset_state()
        neighbor_loader.reset_state()
        with torch.no_grad():
            tgn_memory.memory.data = node_feat_proj(node_feats_th)
        tgn_memory.train()

        m_loss = []
        pbar = tqdm(train_edge_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            batch = batch.to(device)
            src, dst, t, m = batch.src, batch.dst, batch.t, batch.msg

            # find which src nodes in this batch have train labels
            src_np      = src.cpu().numpy()
            label_vals  = train_label_arr[src_np]
            labeled_mask = label_vals >= 0

            if labeled_mask.any() and labeled_mask.sum() >= 2:
                labeled_src    = src[torch.from_numpy(labeled_mask).to(device)]
                label_tensor   = torch.tensor(
                    label_vals[labeled_mask], dtype=torch.float, device=device
                )
                z    = get_node_embeddings(labeled_src)
                pred = node_clf(z).squeeze(-1)
                loss = criterion(pred, label_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m_loss.append(loss.item())
                pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

            # update memory with current batch edges (after classification)
            neighbor_loader.insert(src, dst)
            tgn_memory.update_state(src, dst, t.long(), m)
            tgn_memory.detach()

        # validation — replay train+val edges so val nodes have full context
        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr, val_loss = eval_node_clf(
            val_nodes, val_labels, val_replay_flag
        )

        train_loss = np.mean(m_loss) if m_loss else float('nan')
        logger.info(
            f'Epoch {epoch} | loss: {train_loss:.4f} | val loss: {val_loss:.4f} | '
            f'val auc: {val_auc:.4f} | val ap: {val_ap:.4f} | '
            f'val f1: {val_f1:.4f} | val mcc: {val_mcc:.4f} | '
            f'val recall: {val_rc:.4f} | val precision: {val_pr:.4f}'
        )
        mlflow.log_metrics({
            'loss':          train_loss,
            'val_loss':      val_loss,
            'val_auc':       val_auc,
            'val_ap':        val_ap,
            'val_f1':        val_f1,
            'val_mcc':       val_mcc,
            'val_recall':    val_rc,
            'val_precision': val_pr,
        }, step=epoch)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_auc_hist.append(val_auc)

        if early_stopper.early_stop_check(val_auc):   # use AUC for early stopping
            logger.info(f'Early stopping at epoch {epoch}')
            gnn.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.gnn'))
            node_clf.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.clf'))
            break
        else:
            if early_stopper.best_epoch == epoch:  # only save if this is the new best
                # remove previous best checkpoint
                for suffix in ['.gnn', '.clf']:
                    path = get_checkpoint_path(last_saved_epoch) + suffix
                    if os.path.exists(path):
                        os.remove(path)
                torch.save(gnn.state_dict(),      get_checkpoint_path(epoch) + '.gnn')
                torch.save(node_clf.state_dict(), get_checkpoint_path(epoch) + '.clf')
                last_saved_epoch = epoch

    # final test — replay all edges for full context
    test_replay_flag = train_edge_flag | val_edge_flag | test_edge_flag | nn_test_edge_flag  # replay all train+val+test edges for test evaluation
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, test_loss = eval_node_clf(
        test_nodes, test_labels, edge_flag_for_replay=test_replay_flag
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
        'test_loss':      test_loss,
    })

    torch.save(gnn.state_dict(),      MODEL_SAVE_PATH + '.gnn')
    torch.save(node_clf.state_dict(), MODEL_SAVE_PATH + '.clf')
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')

    # plot training loss and val auc
    epochs = list(range(len(train_loss_hist)))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = '#e05c5c'
    color_auc  = '#5c8de0'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, train_loss_hist, color=color_loss, linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_loss_hist,   color='#e0a85c',  linewidth=2, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val AUC', color=color_auc)
    ax2.plot(epochs, val_auc_hist, color=color_auc, linewidth=2, linestyle='--', label='Val AUC')
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.set_ylim(0, 1)

    # mark best epoch
    best_ep = early_stopper.best_epoch
    ax2.axvline(x=best_ep, color='gray', linestyle=':', linewidth=1.5, label=f'Best epoch ({best_ep})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('TGN Node Classification — Training Curve (early stop on val AP)')
    plt.tight_layout()

    plot_path = f'./saved_models/{args.prefix}-tgn-node-{DATA}-training-curve.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    mlflow.log_artifact(plot_path)
    logger.info(f'Training curve saved to {plot_path}')