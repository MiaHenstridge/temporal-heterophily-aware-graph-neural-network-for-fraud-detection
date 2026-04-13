import pandas as pd
import numpy as np
import os
from namespaces import DA

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DATA_PATH = '../datasets/DGraphFin/'
os.makedirs(DA.paths.data, exist_ok=True)


# OUTPUT_DATA_PATH = '../processed_data'
os.makedirs(DA.paths.output_data, exist_ok=True)

# OUTPUT_DATA_ML = os.path.join(OUTPUT_DATA_PATH, 'baseline_ml')
os.makedirs(DA.paths.output_data_ml, exist_ok=True)


# 1. Load the dataset
data = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))
edge_time = np.load(os.path.join(DA.paths.data, 'dgraphfinv2_edge_timestamp.npy')).astype(np.float32)
node_time = np.load(os.path.join(DA.paths.data, 'dgraphfinv2_node_timestamp.npy')).astype(np.float32)

# 2. Extract node features and labels
x = data['x']                                            # [N, 17]
y = data['y']
N = x.shape[0]

# edge_index from raw npz: stored as [E, 2] with i→j (applicant→contact)
# flip to j→i to match temporal model convention (contact→applicant)
edge_index_raw = data['edge_index']                      # [E, 2]
src_raw = edge_index_raw[:, 1].astype(np.int64)          # j (contact)
dst_raw = edge_index_raw[:, 0].astype(np.int64)          # i (applicant)
E = len(src_raw)

# 3. relabel y for binary classification
print(f"Original classes: {np.unique(y)}")
y = np.where(y == 1, 1, 0)
print(f"Classes after relabeling: {np.unique(y)}")

# 4. Split masks
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask  = data['test_mask']

# ─────────────────────────────────────────────────────────────────────────────
# 5. Engineer structural and temporal features
#    All features are computed from the full graph (transductive setting).
#    Normalisation is fit on training nodes only.
# ─────────────────────────────────────────────────────────────────────────────

print("Computing engineered features...")

# ── degree features ───────────────────────────────────────────────────────────
# fix background node sentinels here too (needed before edge filter below)
max_ts     = edge_time.max()
query_time = node_time.copy()
query_time[query_time < 0] = max_ts

# filter edges to t_edge < query_time[dst] before computing ALL features
# this ensures no future information leaks into any feature
valid_mask_edges = edge_time < query_time[dst_raw]          # [E] bool
src_filtered     = src_raw[valid_mask_edges]
dst_filtered     = dst_raw[valid_mask_edges]
et_filtered      = edge_time[valid_mask_edges]

print(f"  Edges before temporal filter: {E:,}")
print(f"  Edges after  temporal filter: {valid_mask_edges.sum():,}")

# degrees computed from temporally-valid edges only
out_deg   = np.bincount(src_filtered, minlength=N).astype(np.float32)
in_deg    = np.bincount(dst_filtered, minlength=N).astype(np.float32)
deg_ratio = out_deg / (in_deg + 1.0)                              # avoid div-by-zero

# ── temporal features ─────────────────────────────────────────────────────────
# Mirrors _augment_features in THEGCNSamplerModel exactly.
# dts = query_time[dst] - edge_time (larger = older edge), same as sampler output.
# Edges already filtered to t_edge < query_time[dst] in the degree section above.

# compute dts per already-filtered edges
dts = (query_time[dst_filtered] - et_filtered).astype(np.float32)  # [E_valid]

# sort edges by dst for per-node aggregation
sort_order  = np.argsort(dst_filtered, kind='stable')
dst_sorted  = dst_filtered[sort_order]
dts_sorted  = dts[sort_order]

node_edge_count = np.bincount(dst_filtered, minlength=N).astype(np.float32)

# recency: min dts per dst (smallest dts = most recent edge)
# init with 1e9 sentinel, then clamp to 1e8 — mirrors INF.clone().clamp(max=1e8)
recency = np.full(N, 1e9, dtype=np.float32)
np.minimum.at(recency, dst_sorted, dts_sorted)
recency = recency.clip(max=1e8)                             # cap sentinels for nodes with no edges

# max dts per dst (oldest edge)
max_dts_per_node = np.zeros(N, dtype=np.float32)
np.maximum.at(max_dts_per_node, dst_sorted, dts_sorted)

# burst ratio: fraction of edges with dts <= min_dts + 0.25 * activity_window
# anchored at min_dts (most recent) — matches: burst_cutoff = min_dts + 0.25 * window
activity_window = (max_dts_per_node - recency).clip(min=1.0)
burst_cutoff    = recency + 0.25 * activity_window          # [N]
is_burst        = (dts_sorted <= burst_cutoff[dst_sorted]).astype(np.float32)
burst_count     = np.zeros(N, dtype=np.float32)
np.add.at(burst_count, dst_sorted, is_burst)
burst_ratio = burst_count / node_edge_count.clip(min=1.0)

# mean_dts and std_dts per dst node (temporal diversity of neighbourhood)
mean_dts = np.zeros(N, dtype=np.float32)
np.add.at(mean_dts, dst_sorted, dts_sorted)
mean_dts = mean_dts / node_edge_count.clip(min=1.0)

dts_sq   = np.zeros(N, dtype=np.float32)
np.add.at(dts_sq, dst_sorted, dts_sorted ** 2)
dts_sq   = dts_sq / node_edge_count.clip(min=1.0)
std_dts  = np.sqrt((dts_sq - mean_dts ** 2).clip(min=0))

# ── stack all engineered features — matches N_EXTRA=7 in THEGCNSamplerModel ──
extra = np.stack([
    out_deg, in_deg, deg_ratio,
    recency, burst_ratio, mean_dts, std_dts,
], axis=1).astype(np.float32)                            # [N, 7]

print(f"  Engineered features shape: {extra.shape}")

# ── normalise: fit on train, apply to all ─────────────────────────────────────
train_extra = extra[train_mask]
feat_mean   = train_extra.mean(axis=0)
feat_std    = train_extra.std(axis=0).clip(min=1e-8)
extra_norm  = (extra - feat_mean) / feat_std

# ── concatenate with base features ───────────────────────────────────────────
x_augmented = np.concatenate([x, extra_norm], axis=1)   # [N, 17+7=24]
print(f"  Augmented feature matrix shape: {x_augmented.shape}")

# ── split ─────────────────────────────────────────────────────────────────────
x_train, y_train = x_augmented[train_mask], y[train_mask]
x_val,   y_val   = x_augmented[valid_mask], y[valid_mask]
x_test,  y_test  = x_augmented[test_mask],  y[test_mask]

print(f"Total nodes:     {N:,}")
print(f"Fraud nodes:     {np.sum(y==1):,}")
print(f"Non-fraud nodes: {np.sum(y==0):,}")
print(f"Train: {x_train.shape[0]:,} | Val: {x_val.shape[0]:,} | Test: {x_test.shape[0]:,}")

# 6. save to a single compressed file
np.savez_compressed(
    os.path.join(DA.paths.output_data_ml, 'dgraphfin_processed.npz'),
    x_train=x_train, y_train=y_train,
    x_val=x_val,     y_val=y_val,
    x_test=x_test,   y_test=y_test,
    feat_mean=feat_mean,   # save scaler params for inference
    feat_std=feat_std,
)
print("Saved to dgraphfin_processed.npz")