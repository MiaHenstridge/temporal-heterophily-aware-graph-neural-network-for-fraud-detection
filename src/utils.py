import numpy as np
import pandas as pd
import torch
import dgl
import torch.nn.functional as F
# from scipy.sparse import csr_matrix
# from dgraphfin import load_dgraphfin_temporal
# import sampler_core

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


# ─────────────────────────────────────────────────────────────────────────────
# Static graph feature augmentation
# ─────────────────────────────────────────────────────────────────────────────
 
def augment_static_features(x, edge_index, train_idx):
    """
    Compute degree-based structural features from the full directed graph
    and concatenate with base node features.
 
    Parameters
    ----------
    x          : FloatTensor [N, F]    base node features (on any device)
    edge_index : LongTensor  [2, E]    DIRECTED edge index (src -> dst)
                                        use the raw directed edges before
                                        symmetrisation for meaningful
                                        out_deg / in_deg separation
    train_idx  : LongTensor  [N_train] training node indices
 
    Returns
    -------
    FloatTensor [N, F+3]   base features concatenated with normalised
                            [out_deg, in_deg, deg_ratio]
    """
    N      = x.shape[0]
    device = x.device
    src, dst = edge_index[1].to(device), edge_index[0].to(device)  # flip to j -> i: contact -> applicant
 
    ones      = torch.ones(src.shape[0], device=device)
    out_deg   = torch.zeros(N, device=device).scatter_add_(0, src, ones)
    in_deg    = torch.zeros(N, device=device).scatter_add_(0, dst, ones)
    deg_ratio = out_deg / (in_deg + 1.0)
 
    extra = torch.stack([out_deg, in_deg, deg_ratio], dim=1).float()  # [N, 3]
 
    # fit normalisation on training nodes only
    mean  = extra[train_idx].mean(0)
    std   = extra[train_idx].std(0).clamp(min=1e-8)
    extra = (extra - mean) / std
 
    return torch.cat([x, extra], dim=1)   # [N, F+3]


# ─────────────────────────────────────────────────────────────────────────────
# Load temporal graph for Temporal Parallel Sampler
# ─────────────────────────────────────────────────────────────────────────────
def load_graph(data_dir):
    df = pd.read_csv(f'{data_dir}/edges.csv')
    g = np.load(f'{data_dir}/ext_full.npz')
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




