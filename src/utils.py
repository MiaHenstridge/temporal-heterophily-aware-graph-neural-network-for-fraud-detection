import numpy as np
import pandas as pd
import torch
import dgl
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






