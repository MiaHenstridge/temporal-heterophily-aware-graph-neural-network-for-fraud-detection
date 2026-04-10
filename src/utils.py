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


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary Focal Loss
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
        Args:
            alpha (float): Balancing factor for the rare class (fraud).
            gamma (float): Focusing parameter. Higher values focus more on hard samples.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits, targets are 0 or 1
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # p_t is the probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
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


# load tgl graph (edges.csv and ext_full.npz)
def load_graph(data_dir):
    df = pd.read_csv(f'{data_dir}/edges.csv')
    g = np.load('{data_dir}/ext_full.npz')
    return g, df






