import pandas as pd
import argparse
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
from namespaces import DA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

######### load data ##################
data = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))
# edge timestamps from DGraphFinv2 dataset
edge_time = np.load(os.path.join(DA.paths.data, 'dgraphfinv2_edge_timestamp.npy'))                                                 
# node timestamps from DGraphFinv2 dataset
node_time = np.load(os.path.join(DA.paths.data, 'dgraphfinv2_node_timestamp.npy'))                                                 

######## prepare edges.csv ############
df = pd.DataFrame(
    {
        'src': data['edge_index'][:, 0],
        'dst': data['edge_index'][:, 1],
        'time': edge_time.astype(np.float32)
    }
)

def get_ext_roll_for_edges(data, df):
    # 1. Determine the maximum node ID present in the edges
    # This ensures our masks are large enough to be indexed by any src/dst
    max_node_id = max(df['src'].max(), df['dst'].max())
    num_nodes = max_node_id + 1
    
    # 2. Initialize global boolean masks with False
    full_train = np.zeros(num_nodes, dtype=bool)
    full_valid = np.zeros(num_nodes, dtype=bool)
    full_test  = np.zeros(num_nodes, dtype=bool)
    full_bg    = np.zeros(num_nodes, dtype=bool)

    # 3. Fill the masks using the indices provided in 'data'
    full_train[data['train_mask']] = True
    full_valid[data['valid_mask']] = True
    full_test[data['test_mask']]  = True

    # 4. Handle Background Nodes (y not in [0, 1])
    # We only check 'y' for nodes that actually have a label defined
    y_labels = data['y']
    n_labels = len(y_labels)
    # Nodes with labels that aren't 0 or 1
    bg_indices = np.where(~np.isin(y_labels, [0, 1]))[0]
    full_bg[bg_indices] = True
    
    # Also treat any node beyond the label array as "Background"
    if num_nodes > n_labels:
        full_bg[n_labels:] = True

    # 5. Map to the DataFrame (Standard Vectorization)
    src, dst = df['src'].values, df['dst'].values
    
    s_train, d_train = full_train[src], full_train[dst]
    s_valid, d_valid = full_valid[src], full_valid[dst]
    s_test,  d_test  = full_test[src],  full_test[dst]
    s_bg,    d_bg    = full_bg[src],    full_bg[dst]

    # 6. Define the ext_roll conditions
    m0 = (s_train & d_train) | (s_train & d_bg) | (d_train & s_bg) | (s_bg & d_bg)
    m1 = (s_valid & d_valid) | (s_valid & (d_train | d_bg)) | (d_valid & (s_train | s_bg))
    m2 = (s_test | d_test)

    # Use np.select to assign the rolls
    df['ext_roll'] = np.select([m0, m1, m2], [0, 1, 2], default=-1)
    
    return df

df = get_ext_roll_for_edges(data, df)
# sort by time ascending
df.sort_values(by='time', ascending=True, inplace=True)

# write to destination path
print("saving edges.csv...")
df.to_csv(os.path.join(DA.paths.output_data_tgl, 'edges.csv'))


########## Prepare ext_full.npz: The T-CSR representation of the temporal graph #############
parser=argparse.ArgumentParser()
parser.add_argument('--add_reverse', default=False, action='store_true')
args=parser.parse_args()

df = pd.read_csv(os.path.join(DA.paths.output_data_tgl, 'edges.csv'))
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_train_indices = [[] for _ in range(num_nodes)]
int_train_ts = [[] for _ in range(num_nodes)]
int_train_eid = [[] for _ in range(num_nodes)]

int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_full_indices = [[] for _ in range(num_nodes)]
int_full_ts = [[] for _ in range(num_nodes)]
int_full_eid = [[] for _ in range(num_nodes)]

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    if row['ext_roll'] == 0:
        int_train_indices[src].append(dst)
        int_train_ts[src].append(row['time'])
        int_train_eid[src].append(idx)
        if args.add_reverse:
            int_train_indices[dst].append(src)
            int_train_ts[dst].append(row['time'])
            int_train_eid[dst].append(idx)
        # int_train_indptr[src + 1:] += 1
    if row['ext_roll'] != 3:
        int_full_indices[src].append(dst)
        int_full_ts[src].append(row['time'])
        int_full_eid[src].append(idx)
        if args.add_reverse:
            int_full_indices[dst].append(src)
            int_full_ts[dst].append(row['time'])
            int_full_eid[dst].append(idx)
        # int_full_indptr[src + 1:] += 1
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row['time'])
        ext_full_eid[dst].append(idx)
    # ext_full_indptr[src + 1:] += 1

for i in tqdm(range(num_nodes)):
    int_train_indptr[i + 1] = int_train_indptr[i] + len(int_train_indices[i])
    int_full_indptr[i + 1] = int_full_indptr[i] + len(int_full_indices[i])
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

int_train_indices = np.array(list(itertools.chain(*int_train_indices)))
int_train_ts = np.array(list(itertools.chain(*int_train_ts)))
int_train_eid = np.array(list(itertools.chain(*int_train_eid)))

int_full_indices = np.array(list(itertools.chain(*int_full_indices)))
int_full_ts = np.array(list(itertools.chain(*int_full_ts)))
int_full_eid = np.array(list(itertools.chain(*int_full_eid)))

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')
def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]

for i in tqdm(range(int_train_indptr.shape[0] - 1)):
    tsort(i, int_train_indptr, int_train_indices, int_train_ts, int_train_eid)
    tsort(i, int_full_indptr, int_full_indices, int_full_ts, int_full_eid)
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

# import pdb; pdb.set_trace()
print('saving ext_full.npz')
np.savez(os.path.join(DA.paths.output_data_tgl, 'int_train.npz'), indptr=int_train_indptr, indices=int_train_indices, ts=int_train_ts, eid=int_train_eid)
np.savez(os.path.join(DA.paths.output_data_tgl, 'int_full.npz'), indptr=int_full_indptr, indices=int_full_indices, ts=int_full_ts, eid=int_full_eid)
np.savez(os.path.join(DA.paths.output_data_tgl, 'ext_full.npz'), indptr=ext_full_indptr, indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)