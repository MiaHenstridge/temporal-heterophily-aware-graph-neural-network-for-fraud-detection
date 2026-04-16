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

from torch_geometric.loader import NeighborLoader
from models import GraphSAGEModel, GATModel, GATv2Model, FAGCNModel

import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from namespaces import DA
from dgraphfin import load_dgraphfin

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(DA.paths.log, exist_ok=True)
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)

RANDOM_SEED = 1111

set_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser('Static GNN Node Classification (GraphSAGE / GAT / GATv2)')
parser.add_argument('-d', '--data',      type=str,   default='DGraphFin')
parser.add_argument('--data_dir',        type=str,   default='./datasets',
                    help='root directory containing raw/dgraphfin.npz')
parser.add_argument('--model',           type=str,   default='sage',
                    choices=['sage', 'gat', 'gatv2', 'fagcn'],
                    help='sage = GraphSAGE, gat = GAT, gatv2 = GATv2, fagcn = FAGCN')
parser.add_argument('--bs',              type=int,   default=1024,
                    help='number of seed nodes per mini-batch')
parser.add_argument('--n_epoch',         type=int,   default=100)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--drop_out',        type=float, default=0.2)
parser.add_argument('--gpu',             type=int,   default=0)
parser.add_argument('--n_layer',         type=int,   default=2)
parser.add_argument('--node_dim',        type=int,   default=128)
parser.add_argument('--feat_augment',     action='store_true', default=False,
                    help='whether to augment features')
parser.add_argument('--n_head',           type=int,   default=4,
                    help='number of attention heads (GAT / GATv2 only)')
parser.add_argument('--eps',             type=float, default=0.1,
                    help='initial value of residual weight (FAGCN only)')
parser.add_argument('--n_neighbor',      type=int,   default=10,
                    help='neighbors sampled per layer in NeighborLoader')
parser.add_argument('--fold',            type=int,   default=0,
                    help='which fold to use when dataset has multiple splits')
parser.add_argument('--prefix',          type=str,   default='')
parser.add_argument('--max_round',       type=int,   default=10)
parser.add_argument('--tolerance',       type=float, default=1e-4)

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

parser.add_argument('--num_workers',     type=int,   default=12)
parser.add_argument('--weight_decay',    type=float, default=5e-7)
parser.add_argument('--early_stop_higher_better', action='store_true', default=False)

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
FEAT_AUGMENT  = args.feat_augment
NUM_LAYER     = args.n_layer
NUM_NEIGHBOR  = args.n_neighbor
MAX_ROUND     = args.max_round
TOLERANCE     = args.tolerance
MODEL_TYPE    = args.model
LOSS          = args.loss
NUM_WORKERS   = args.num_workers
WEIGHT_DECAY  = args.weight_decay
EARLY_STOP_HIGHER_BETTER = args.early_stop_higher_better

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Load graph data (for node features, labels, masks, node_time)
# ─────────────────────────────────────────────────────────────────────────────
bundle = load_dgraphfin(data_dir=args.data_dir, fold=args.fold, to_undirected=False)
et = torch.tensor(np.load(os.path.join(args.data_dir, args.data, 'dgraphfinv2_edge_timestamp.npy')), dtype=torch.float32)
nt = torch.tensor(np.load(os.path.join(args.data_dir, args.data, 'dgraphfinv2_node_timestamp.npy')), dtype=torch.float32)

graph         = bundle.graph
train_idx     = bundle.train_idx
val_idx       = bundle.val_idx
test_idx      = bundle.test_idx
train_labels_np = bundle.train_labels
node_feat_dim = bundle.node_feat_dim

if FEAT_AUGMENT:
    graph.x       = augment_static_features(graph.x, graph.edge_index, train_idx, et, nt)
    node_feat_dim = graph.x.shape[1]

logger.info(f'Train nodes: {len(train_idx)} | Val nodes: {len(val_idx)} | Test nodes: {len(test_idx)}')
logger.info(f'Train fraud rate: {train_labels_np.mean():.4f}')
logger.info(f'Graph: {graph.num_nodes} nodes | {graph.num_edges} edges')

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# NeighborLoader
# ─────────────────────────────────────────────────────────────────────────────
# num neighbors per layer
if NUM_LAYER == 1:
    num_neighbors = [NUM_NEIGHBOR]
else:
    num_neighbors = [NUM_NEIGHBOR] * NUM_LAYER

logger.info(f'NeighborLoader fanouts (outer→inner): {num_neighbors}')

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

if MODEL_TYPE == 'sage':
    model = GraphSAGEModel(
        in_channels     = node_feat_dim,
        hidden_channels = NODE_DIM,
        n_layers        = NUM_LAYER,
        dropout         = DROP_OUT,
    ).to(device)
elif MODEL_TYPE == 'gat':
    model = GATModel(
        in_channels     = node_feat_dim,
        hidden_channels = NODE_DIM,
        n_layers        = NUM_LAYER,
        heads           = args.n_head,
        dropout         = DROP_OUT,
    ).to(device)
elif MODEL_TYPE == 'gatv2':  # gatv2
    model = GATv2Model(
        in_channels     = node_feat_dim,
        hidden_channels = NODE_DIM,
        n_layers        = NUM_LAYER,
        heads           = args.n_head,
        dropout         = DROP_OUT,
    ).to(device)
else:                   #fagcn
    model = FAGCNModel(
        in_channels=node_feat_dim,
        hidden_channels=NODE_DIM,
        n_layers=NUM_LAYER,
        dropout=DROP_OUT,
        eps=args.eps,
    ).to(device)

logger.info(f'Model: {MODEL_TYPE.upper()} | params: {sum(p.numel() for p in model.parameters()):,}')

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
def eval_nodes(loader, criterion):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch      = batch.to(device, non_blocking=True)
        batch_size = batch.batch_size

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

    val_loss = criterion(
        torch.tensor(all_preds, device=device),
        torch.tensor(all_labels, device=device),
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

train_loss_hist = []
val_loss_hist   = []
val_auc_hist    = []

# create experiment
EXPERIMENT_NAME = f'static-gnn-{MODEL_TYPE}'
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_SAVE_PATH     = f'./saved_models/{EXPERIMENT_NAME}-{args.prefix}-{DATA}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{EXPERIMENT_NAME}-{args.prefix}-{DATA}.pt'

with mlflow.start_run():
    mlflow.log_params(vars(args))

    for epoch in range(NUM_EPOCH):
        model.train()
        m_loss = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            batch      = batch.to(device, non_blocking=True)
            batch_size = batch.batch_size

            pred        = model(batch.x, batch.edge_index, batch_size)
            label_batch = batch.y[:batch_size]

            loss = criterion(pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            m_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(m_loss):.4f}'})

        val_auc, val_ap, val_f1, val_mcc, val_rc, val_pr, val_loss = eval_nodes(val_loader, criterion)
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
            model.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch) + '.pt'))
            break
        else:
            if early_stopper.best_epoch == epoch:
                prev = get_checkpoint_path(last_saved_epoch) + '.pt'
                if os.path.exists(prev):
                    os.remove(prev)
                torch.save(model.state_dict(), get_checkpoint_path(epoch) + '.pt')
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


    # ── final test ──────────────────────────────────────────────────────────
    # Reload the best checkpoint into the model before testing and final saving
    logger.info(f"Loading best model from epoch {early_stopper.best_epoch} for testing.")
    model.load_state_dict(torch.load(get_checkpoint_path(early_stopper.best_epoch)))
    
    test_auc, test_ap, test_f1, test_mcc, test_rc, test_pr, _ = eval_nodes(test_loader, criterion)
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

    # # ── training curve plot ─────────────────────────────────────────────────
    # epochs_list = list(range(len(train_loss_hist)))
    # fig, ax1 = plt.subplots(figsize=(10, 5))

    # color_loss_train = '#e05c5c'
    # color_loss_val   = '#e09c5c'
    # color_auc        = '#5c8de0'

    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Train Loss', color=color_loss_train)
    # ax1.plot(epochs_list, train_loss_hist, color=color_loss_train, linewidth=2, label='Train Loss')
    # ax1.plot(epochs_list, val_loss_hist,   color=color_loss_val,   linewidth=2, linestyle='--', label='Val Loss')
    # ax1.tick_params(axis='y', labelcolor=color_loss_train)

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Val AUC', color=color_auc)
    # ax2.plot(epochs_list, val_auc_hist, color=color_auc, linewidth=2, linestyle='--', label='Val AUC')
    # ax2.tick_params(axis='y', labelcolor=color_auc)
    # ax2.set_ylim(0, 1)

    # best_ep = early_stopper.best_epoch
    # ax2.axvline(x=best_ep, color='gray', linestyle=':', linewidth=1.5, label=f'Best epoch ({best_ep})')

    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # plt.title(f'{MODEL_TYPE.upper()} Node Classification — Training Curve (early stop on val loss)')
    # plt.tight_layout()

    # plot_path = f'./saved_models/{EXPERIMENT_NAME}-{args.prefix}-{DATA}-training-curve.png'
    # plt.savefig(plot_path, dpi=150)
    # plt.close()
    # mlflow.log_artifact(plot_path)
    # logger.info(f'Training curve saved to {plot_path}')