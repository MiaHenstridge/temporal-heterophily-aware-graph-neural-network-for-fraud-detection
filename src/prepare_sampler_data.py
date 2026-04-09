"""
prepare_sampler_data.py
=======================
Converts DGraphFin temporal graph data into the CSC format required by the
TGL C++ ParallelSampler, and saves it as a .npz file.
 
The C++ sampler requires:
  - Destination-indexed CSC (column = destination node).
  - Within each destination node's slice, edges sorted by timestamp ascending.
 
This is a one-time preprocessing step. The output is saved to:
    ./processed_data/tgl/dgraphfin_tgl.npz
 
Run
---
    python src/prepare_sampler_data.py
"""
import os
import sys
import numpy as np
import torch
import torch_geometric.transforms as T
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgraphfin import DGraphFin

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
 
DATA_DIR  = './datasets'
OUT_DIR   = './processed_data/tgl'
OUT_FILE  = os.path.join(OUT_DIR, 'dgraphfin_tgl.npz')
 
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load raw DGraphFin
# ─────────────────────────────────────────────────────────────────────────────
 
print("Loading DGraphFin dataset...")
dataset = DGraphFin(
    root          = os.path.join(DATA_DIR, 'DGraphFin'),
    pre_transform = T.ToSparseTensor(),
)
data = dataset[0]


# Recover directed edge_index from adj_t (row=dst, col=src in adj_t)
row_t, col_t, _ = data.adj_t.coo()
src_directed = col_t.numpy().astype(np.int32)
dst_directed = row_t.numpy().astype(np.int32)

# load edge timestamps from DGraphFinv2
ts_path = os.path.join(DATA_DIR, 'DGraphFin', 'dgraphfinv2_edge_timestamp.npy')
if not os.path.exists(ts_path):
    raise FileNotFoundError(
        f"Edge timestamp file not found: {ts_path}\n"
        "Ensure dgraphfinv2_edge_timestamp.npy is in the DGraphFin dataset directory."
    )
ts_raw = np.load(ts_path).astype(np.float32)    # [E]

num_nodes = int(data.num_nodes)
num_edges = int(data.num_edges)


print(f"  Nodes: {num_nodes:,}  |  Edges: {num_edges:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Build CSC: destination-indexed, sorted by ts within each dst's slice
#
# The C++ sampler is destination-indexed (CSC where column = destination):
#   indptr[n]   : start of edges pointing INTO node n
#   indptr[n+1] : end
#   indices[i]  : source node of edge i (within dst n's slice)
#   eid[i]      : original edge id
#   ts[i]       : edge timestamp
#
# Within each node's slice, edges must be sorted by timestamp ascending so
# that binary search (upper_bound) works correctly in C++.
# ─────────────────────────────────────────────────────────────────────────────

print("Building CSC (destination-indexed, ts-sorted)...")

# sort all edges by (dst, ts) - primary dst, secondary ts ascending
sort_key = dst_directed.astype(np.int64) * (int(ts_raw.max()) + 1) + ts_raw.astype(np.int64)
sort_order = np.argsort(sort_key, kind='stable')

src_sorted = src_directed[sort_order]       # [E] source node ids
dst_sorted = dst_directed[sort_order]       # [E] destination node ids
ts_sorted  = ts_raw[sort_order]             # [E]  timestamps (ascending within each dst)
eid_sorted = sort_order.astype(np.int32)    # [E] original edge ids


# build indptr: count edges per destination node
counts = np.bincount(dst_sorted, minlength=num_nodes)       # [N]
indptr = np.zeros(num_nodes + 1, dtype=np.int32)
np.cumsum(counts, out=indptr[1:])

# sanity check
assert indptr[-1] == num_edges, f"indptr[-1]={indptr[-1]} != num_edges={num_edges}"

# verify ts is sorted within each dst slice (sample check)
print("  Verifying timestamp ordering within destination slices...")
n_check = min(10000, num_nodes)
check_nodes = np.random.default_rng(42).integers(0, num_nodes, n_check)
for n in check_nodes:
    s, e = indptr[n], indptr[n + 1]
    if e - s > 1:
        assert np.all(np.diff(ts_sorted[s:e]) >= 0), \
            f"Timestamps not sorted for node {n}: {ts_sorted[s:e]}"
print(f"  OK (checked {n_check:,} random nodes)")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
 
print(f"Saving to {OUT_FILE}...")
np.savez(
    OUT_FILE,
    indptr    = indptr,
    indices   = src_sorted,
    eid       = eid_sorted,
    ts        = ts_sorted,
    num_nodes = np.array(num_nodes),
    num_edges = np.array(num_edges),
)
print(f"Done. File size: {os.path.getsize(OUT_FILE) / 1e6:.1f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity print
# ─────────────────────────────────────────────────────────────────────────────
 
print("\nSummary:")
print(f"  indptr  shape : {indptr.shape}    dtype: {indptr.dtype}")
print(f"  indices shape : {src_sorted.shape}  dtype: {src_sorted.dtype}")
print(f"  eid     shape : {eid_sorted.shape}  dtype: {eid_sorted.dtype}")
print(f"  ts      shape : {ts_sorted.shape}  dtype: {ts_sorted.dtype}")
print(f"  ts range      : [{ts_sorted.min():.1f}, {ts_sorted.max():.1f}]")
print(f"  avg degree    : {num_edges / num_nodes:.1f} edges/node")
 