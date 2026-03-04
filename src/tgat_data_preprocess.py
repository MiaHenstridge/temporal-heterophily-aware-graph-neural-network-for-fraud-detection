import numpy as np
import pandas as pd
import os
from namespaces import DA
import matplotlib.pyplot as plt

os.chdir('/home/mai/notebooks/final_thesis/')
os.getcwd()

os.makedirs(DA.paths.output_data_graph, exist_ok=True)


def process_data(data_path, output_path, data_name='dgraphfin'):
    data = np.load(data_path)
    
    x = data['x']
    y = data['y']
    edge_index = data['edge_index']
    edge_type = data['edge_type']
    edge_timestamp = data['edge_timestamp']
    
    n_nodes = x.shape[0]
    n_edges = edge_index.shape[0]

    # Short edges by timestamp
    sorted_order = np.argsort(edge_timestamp)
    edge_index = edge_index[sorted_order]
    edge_type = edge_type[sorted_order]

    # Build csv file: u, i, ts, label, idx
    # node idx are 1-based for
    u = edge_index[:, 0] + 1
    i = edge_index[:, 1] + 1
    ts = edge_timestamp
    label = np.zeros(n_edges, dtype=int)  # Assuming all edges are unknown
    idx = np.arange(1, n_edges+1)

    df = pd.DataFrame({
        'u': u,
        'i': i,
        'ts': ts,
        'label': label,
        'idx': idx
    })
    df.to_csv(os.path.join(output_path, f'ml_{data_name}.csv'), index=False)
    print(f"Saved ml_{data_name}.csv — shape: {df.shape}")

    # edge features: [n_edges+1, edge_feat_dim]
    # use one-hot encoding for edge type
    # row 0 is the null/padding embedding
    n_edge_types = int(np.max(edge_type)) + 1
    edge_feats = np.zeros((n_edges + 1, n_edge_types), dtype=np.float32)
    edge_feats[1:, :] = np.eye(n_edge_types)[edge_type]   # one-hot
    np.save(os.path.join(output_path, f'ml_{data_name}.npy'), edge_feats)
    print(f"Saved ml_{data_name}.npy — shape: {edge_feats.shape}")

    # node features: [n_nodes+1, node_feat_dim]
    # row 0 is the null/padding embedding
    node_feats = np.zeros((n_nodes + 1, x.shape[1]), dtype=np.float32)
    node_feats[1:, :] = x

    np.save(os.path.join(output_path, f'ml_{data_name}_node.npy'), node_feats)
    print(f"Saved ml_{data_name}_node.npy — shape: {node_feats.shape}")

    return


if __name__ == "__main__":
    process_data(os.path.join(DA.paths.data, 'dgraphfin.npz'), DA.paths.output_data_graph)