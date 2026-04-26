import numpy as np
import pandas as pd
import torch
import dgl
import torch.nn.functional as F
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            self.best_epoch = self.epoch_count
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1
        return self.num_round >= self.max_round
    

# Balanced Focal Loss function
class BalancedFocalLoss(torch.nn.Module):
    """
    Balanced Focal Loss for binary classification with severe class imbalance.
 
    Formula (Lin et al., 2017):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 
    where:
        p_t = sigmoid(logit)      for positive labels (y=1)
        p_t = 1 - sigmoid(logit)  for negative labels (y=0)
        alpha_t = alpha     for positives
        alpha_t = 1 - alpha for negatives
 
    Compared to BCEWithLogitsLoss(pos_weight=w):
      - pos_weight linearly scales the positive loss.
      - Focal loss additionally suppresses easy negatives via (1-p_t)^gamma,
        which is more effective when the model already assigns low scores to
        most negatives (as happens after a few epochs on DGraphFin).
 
    Parameters
    ----------
    alpha : float
        Weight for the positive (fraud) class. Should be > 0.5 when positives
        are rare. Typical values: 0.75 – 0.95 for 1% fraud rate.
        alpha=0.5 reduces to unweighted focal loss.
    gamma : float
        Focusing exponent. gamma=0 reduces to standard cross-entropy.
        gamma=2 is the value used in the original paper (RetinaNet).
        Higher gamma → stronger suppression of easy examples.
        Typical values for fraud detection: 1.0 – 3.0.
    reduction : str
        'mean' (default), 'sum', or 'none'.
    """
 
    def __init__(
        self,
        alpha:     float = 0.75,
        gamma:     float = 2.0,
        reduction: str   = 'mean',
    ):
        super().__init__()
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        assert gamma >= 0,    "gamma must be >= 0"
        assert reduction in ('mean', 'sum', 'none')
 
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
 
    def forward(
        self,
        logits: torch.Tensor,   # [N]  raw model output (before sigmoid)
        labels: torch.Tensor,   # [N]  binary float labels (0.0 or 1.0)
    ) -> torch.Tensor:
 
        # Numerically stable BCE per element: log(sigmoid(x)) and log(1-sigmoid(x))
        # Using F.binary_cross_entropy_with_logits with reduction='none'
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )                                                   # [N]
 
        # p_t: probability of the true class
        #   for y=1: p_t = sigmoid(logit)
        #   for y=0: p_t = 1 - sigmoid(logit) = sigmoid(-logit)
        probs = torch.sigmoid(logits)                       # [N]
        p_t   = labels * probs + (1.0 - labels) * (1.0 - probs)  # [N]
 
        # Focal weight: (1 - p_t)^gamma
        # Easy examples (p_t → 1) get weight → 0; hard examples keep full weight
        focal_weight = (1.0 - p_t) ** self.gamma           # [N]
 
        # Alpha balancing: alpha for positives, (1-alpha) for negatives
        alpha_t = labels * self.alpha + (1.0 - labels) * (1.0 - self.alpha)  # [N]
 
        # Combined loss
        loss = alpha_t * focal_weight * bce                # [N]
 
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
 
    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction}'

    
# function for recall@topk%
def recall_at_top_n_percent(y_true, y_scores, n_percent):
    """
    Calculates Recall @ Top N% of predictions.
    Ensures inputs are 1D to avoid dimensionality errors.
    """
    # Force inputs to be 1D arrays
    y_true = np.array(y_true).ravel()
    y_scores = np.array(y_scores).ravel()
    
    # Create a DataFrame for easy sorting
    df = pd.DataFrame({'true': y_true, 'score': y_scores})
    
    # 1. Sort by score in descending order
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # 2. Determine the cutoff index k
    n_rows = len(df)
    k = int(np.ceil((n_percent / 100.0) * n_rows))
    
    # 3. Identify positives in the top k and total positives
    tp_k = df['true'].iloc[:k].sum()
    total_positives = df['true'].sum()
    
    if total_positives == 0:
        return 0.0
        
    return tp_k / total_positives


def precision_at_top_n_percent(y_true, y_scores, n_percent):
    """
    Calculates Precision @ Top N% of predictions.
    Ensures inputs are 1D to avoid dimensionality errors.
    """
    # Force inputs to be 1D arrays
    y_true = np.array(y_true).ravel()
    y_scores = np.array(y_scores).ravel()
    
    # Create a DataFrame for easy sorting
    df = pd.DataFrame({'true': y_true, 'score': y_scores})
    
    # 1. Sort by score in descending order
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # 2. Determine the cutoff index k
    n_rows = len(df)
    k = int(np.ceil((n_percent / 100.0) * n_rows))
    
    # 3. Identify positives in the top k and predictions in top k
    tp_k = df['true'].iloc[:k].sum()
    total_predictions_k = k
    
    if total_predictions_k == 0:
        return 0.0
        
    return tp_k / total_predictions_k



# ─────────────────────────────────────────────────────────────────────────────
# Static graph feature augmentation
# ─────────────────────────────────────────────────────────────────────────────
 
def augment_static_features(x, edge_index, train_idx, edge_time=None, node_time=None):
    """
    Compute structural and temporal features from the full directed graph
    and concatenate with base node features.
 
    Mirrors the feature set used in THEGCNSamplerModel._augment_features
    so static and temporal GNNs are compared on equal footing.
 
    When edge_time and node_time are provided, a temporal filter is applied
    so only edges with edge_time < node_time[dst] contribute to each node's
    features — matching the TGL sampler's t_edge < t_query constraint.
 
    Parameters
    ----------
    x          : FloatTensor [N, F]
    edge_index : LongTensor  [2, E]   raw directed edge index (i→j,
                                       applicant→contact from npz).
                                       Flipped internally to j→i.
    train_idx  : LongTensor  [N_train]
    edge_time  : LongTensor or FloatTensor [E]   edge timestamps (optional)
    node_time  : LongTensor or FloatTensor [N]   node query timestamps (optional)
 
    Returns
    -------
    FloatTensor [N, F+3]  if edge_time/node_time not provided (degree only)
    FloatTensor [N, F+7]  if edge_time/node_time provided (degree + temporal)
 
    Usage
    -----
    # in 05_static_graph_tuning.py, after load_dgraphfin():
    import numpy as np
    raw    = np.load('./datasets/DGraphFin/dgraphfin.npz')
    ei_dir = torch.tensor(raw['edge_index'], dtype=torch.long).t().contiguous()
    et     = torch.tensor(np.load('...dgraphfinv2_edge_timestamp.npy'), dtype=torch.float)
    nt     = torch.tensor(np.load('...dgraphfinv2_node_timestamp.npy'), dtype=torch.float)
    graph.x      = augment_static_features(graph.x, ei_dir, train_idx, et, nt)
    node_feat_dim = graph.x.shape[1]
    """
    N      = x.shape[0]
    device = x.device
 
    # flip i→j to j→i (contact→applicant) to match temporal model convention
    src = edge_index[1].to(device)   # j (contact)
    dst = edge_index[0].to(device)   # i (applicant)
 
    # ── temporal filter ───────────────────────────────────────────────────────
    if edge_time is not None and node_time is not None:
        et = edge_time.to(device).float()
        nt = node_time.to(device).float()
 
        # fix background node sentinels (large negatives → global max ts)
        max_ts = et.max()
        nt     = nt.clone()
        nt[nt < 0] = max_ts
 
        # keep only edges where edge_time < query_time[dst]
        valid = et < nt[dst]
        src   = src[valid]
        dst   = dst[valid]
        et    = et[valid]
 
        # dts = query_time[dst] - edge_time (larger = older, matches sampler)
        dts = (nt[dst] - et).float()   # [E_valid]
 
    # ── structural features ───────────────────────────────────────────────────
    ones      = torch.ones(src.shape[0], device=device)
    out_deg   = torch.zeros(N, device=device).scatter_add_(0, src, ones)
    in_deg    = torch.zeros(N, device=device).scatter_add_(0, dst, ones)
    deg_ratio = out_deg / (in_deg + 1.0)
 
    extra = torch.stack([out_deg, in_deg, deg_ratio], dim=1).float()  # [N, 3]
 
    # ── temporal features (mirrors _augment_features exactly) ─────────────────
    if edge_time is not None and node_time is not None:
        INF = torch.full((N,), 1e9, device=device)
 
        # recency: min dts per dst (smallest = most recent), capped at 1e8
        recency = INF.clone().scatter_reduce_(
            0, dst, dts, reduce='amin', include_self=True
        ).clamp(max=1e8)
 
        # max dts per dst (oldest edge)
        max_dts = torch.zeros(N, device=device).scatter_reduce_(
            0, dst, dts, reduce='amax', include_self=True
        )
 
        # burst ratio: fraction of edges in most recent 25% of activity window
        activity_window = (max_dts - recency).clamp(min=1.0)
        burst_cutoff    = recency + 0.25 * activity_window
        is_burst        = (dts <= burst_cutoff[dst]).float()
        burst_count     = torch.zeros(N, device=device).scatter_add_(0, dst, is_burst)
        burst_ratio     = burst_count / in_deg.clamp(min=1.0)
 
        # mean and std of dts per dst node
        mean_dts = torch.zeros(N, device=device).scatter_reduce_(
            0, dst, dts, reduce='mean', include_self=False
        )
        dts_sq   = torch.zeros(N, device=device).scatter_reduce_(
            0, dst, dts ** 2, reduce='mean', include_self=False
        )
        std_dts  = (dts_sq - mean_dts ** 2).clamp(min=0).sqrt()
 
        temporal = torch.stack(
            [recency, burst_ratio, mean_dts, std_dts], dim=1
        ).float()                                          # [N, 4]
        extra    = torch.cat([extra, temporal], dim=1)    # [N, 7]
 
    # ── normalise on training nodes only ──────────────────────────────────────
    mean  = extra[train_idx].mean(0)
    std   = extra[train_idx].std(0).clamp(min=1e-8)
    extra = (extra - mean) / std
 
    return torch.cat([x, extra], dim=1)   # [N, F+3] or [N, F+7]


# ─────────────────────────────────────────────────────────────────────────────
# Load temporal graph for Temporal Parallel Sampler
# ─────────────────────────────────────────────────────────────────────────────
def load_graph(data_dir, mode='test'):
    df = pd.read_csv(f'{data_dir}/edges.csv')
    if mode =='train':
        g = np.load(f'{data_dir}/int_train.npz')
    elif mode=='val':
        g = np.load(f'{data_dir}/int_full.npz')
    elif mode=='test':
        g = np.load(f'{data_dir}/ext_full.npz')
    else:
        raise ValueError(f"mode must be 'train', 'val', or 'test'")
    return g, df


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing after Temporal Parallel Sampler
# ─────────────────────────────────────────────────────────────────────────────
def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs


def to_pyg_inputs(mfgs, device='cuda'):
    """
    Converts DGL MFGs from to_dgl_blocks into a structure of torch.Tensors.
    
    Args:
        mfgs (list): Nested list from to_dgl_blocks [[layer2, layer1], ...]
        device (str): Target device ('cuda' or 'cpu')
        
    Returns:
        List of lists where each element is a dictionary containing 
        torch.Tensor versions of edge_index, node_ids, and edge_dts.
    """
    pyg_inputs = []
    
    for history_step in mfgs:
        step_data = []
        for block in history_step:
            # 1. Construct Edge Index [2, E]
            # u (source/neighbors), v (destination/targets)
            u, v = block.edges()
            edge_index = torch.stack([u, v], dim=0).to(device)
            
            # 2. Extract and ensure Tensors for all metadata
            node_ids = block.srcdata['ID'].to(device)
            edge_dts = block.edata['dt'].to(device)
            edge_ids = block.edata['ID'].to(device)
            
            # 3. Package as a dictionary of Tensors
            step_data.append({
                'edge_index': edge_index,
                'node_ids': node_ids,
                'edge_dts': edge_dts,
                'edge_ids': edge_ids,
                'size': (block.num_src_nodes(), block.num_dst_nodes())
            })
        pyg_inputs.append(step_data)
        
    return pyg_inputs




