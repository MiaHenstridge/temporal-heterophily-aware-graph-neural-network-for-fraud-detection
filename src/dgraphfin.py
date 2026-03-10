"""
dgraphfin_dataset.py
====================
Dataset class and data-processing pipeline for the DGraphFin fraud-detection
benchmark.

Public API
----------
    DGraphFin          – PyG InMemoryDataset that wraps dgraphfin.npz
    load_dgraphfin     – one-call helper: loads, processes, and returns
                         everything the training script needs

Dataset layout
--------------
    datasets/
        DGraphFin/            ← place dgraphfin.npz here
    processed_data/      
        graph/data.pt         ← created automatically on first run

Node-label convention (from the official README)
-------------------------------------------------
    0  normal users          (labelled, used for training / eval)
    1  fraud  users          (labelled, used for training / eval)
    2  background users      (unlabelled, present in graph for message passing)
    3  background users      (unlabelled, present in graph for message passing)

train / valid / test masks are 0-based index tensors that cover *only* class-0
and class-1 nodes (70 / 15 / 15 random split).  Class-2 and class-3 nodes are
part of the graph and contribute to neighbourhood aggregation but are never
used as seed nodes or evaluated.
"""

import os
from typing import Callable, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset

# ─────────────────────────────────────────────────────────────────────────────
# Return type of load_dgraphfin
# ─────────────────────────────────────────────────────────────────────────────

class DGraphFinBundle(NamedTuple):
    """All tensors and metadata returned by :func:`load_dgraphfin`."""

    # ── graph (for NeighborLoader) ──────────────────────────────────────────
    graph: Data
    """PyG Data with normalised x, binary float y, symmetric edge_index."""

    # ── split index tensors (0-based, class-0 & class-1 nodes only) ─────────
    train_idx: torch.Tensor   # shape [M_tr]
    val_idx:   torch.Tensor   # shape [M_va]
    test_idx:  torch.Tensor   # shape [M_te]

    # ── numpy label arrays aligned to the index tensors above ───────────────
    train_labels: np.ndarray  # float32, shape [M_tr], values in {0, 1}

    # ── scalar metadata ──────────────────────────────────────────────────────
    node_feat_dim: int        # 17 for the standard DGraphFin release


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryDataset
# ─────────────────────────────────────────────────────────────────────────────

class DGraphFin(InMemoryDataset):
    """
    PyG :class:`~torch_geometric.data.InMemoryDataset` that reads
    ``dgraphfin.npz`` from ``datasets/DGraphFin/`` and persists the processed
    graph to ``processed_data/graph/data.pt``.

    On the **first** run :meth:`process` is called automatically; subsequent
    runs skip straight to loading the cached ``.pt`` file.

    Parameters
    ----------
    root:
        Passed as ``datasets/DGraphFin`` — the directory that directly
        contains ``dgraphfin.npz``.
    transform:
        Optional on-the-fly transform applied after loading.
    pre_transform:
        Optional transform applied once during :meth:`process` and saved.
        Pass ``T.ToSparseTensor()`` here to build ``adj_t`` at save time.

    Attributes on ``data``
    ----------------------
    x           float32  [N, 17]   node features (raw, un-normalised)
    y           long     [N]       node labels: 0 normal / 1 fraud /
                                                2 background / 3 background
    edge_index  long     [2, E]    COO directed edge list
    edge_type   long     [E]       edge-type id (11 types, 0-indexed)
    edge_time   long     [E]       desensitised timestamp
    train_mask  long     [M_tr]    0-based node indices (class 0 & 1 train)
    valid_mask  long     [M_va]    0-based node indices (class 0 & 1 val)
    test_mask   long     [M_te]    0-based node indices (class 0 & 1 test)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # InMemoryDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_dir(self) -> str:
        # Raw file lives directly in root (datasets/DGraphFin/),
        # not in a raw/ subfolder.
        return self.root

    @property
    def processed_dir(self) -> str:
        # Processed cache is stored at the project level under processed_data/,
        # independent of which dataset root was passed in.
        return os.path.join(
            os.path.dirname(os.path.dirname(self.root)), 'processed_data'
        )

    @property
    def raw_file_names(self):
        return ['dgraphfin.npz']

    @property
    def processed_file_names(self):
        return ['graph/data.pt']

    def download(self) -> None:
        # No automatic download – the file must already be present at
        # datasets/DGraphFin/dgraphfin.npz.
        pass

    def process(self) -> None:
        raw = np.load(os.path.join(self.raw_dir, 'dgraphfin.npz'))

        x          = torch.tensor(raw['x'],              dtype=torch.float)
        y          = torch.tensor(raw['y'],              dtype=torch.long)
        edge_index = torch.tensor(raw['edge_index'],     dtype=torch.long)
        edge_type  = torch.tensor(raw['edge_type'],      dtype=torch.long)
        edge_time  = torch.tensor(raw['edge_timestamp'], dtype=torch.long)

        # The .npz stores edges as (E, 2); PyG expects (2, E).
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        # Masks are 0-based node-index arrays, *not* boolean vectors.
        train_mask = torch.tensor(raw['train_mask'], dtype=torch.long)
        valid_mask = torch.tensor(raw['valid_mask'], dtype=torch.long)
        test_mask  = torch.tensor(raw['test_mask'],  dtype=torch.long)

        data = Data(
            x          = x,
            y          = y,
            edge_index = edge_index,
            edge_type  = edge_type,
            edge_time  = edge_time,
            train_mask = train_mask,
            valid_mask = valid_mask,
            test_mask  = test_mask,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


# ─────────────────────────────────────────────────────────────────────────────
# One-call processing helper
# ─────────────────────────────────────────────────────────────────────────────

def load_dgraphfin(data_dir: str, fold: int = 0) -> DGraphFinBundle:
    """
    Load DGraphFin from *data_dir*, apply all processing steps from the
    reference repo (``gnn_mini_batch.py``), and return a
    :class:`DGraphFinBundle` ready for use with
    :class:`~torch_geometric.loader.NeighborLoader`.

    Processing pipeline
    -------------------
    1. **Load** via :class:`DGraphFin` (``T.ToSparseTensor()`` applied once
       at save time so ``adj_t`` is cached in ``processed_data/graph/data.pt``).
    2. **Symmetrise** adjacency: ``data.adj_t = data.adj_t.to_symmetric()``
       — adds reverse edges so message passing is bidirectional.
    3. **Z-score normalise** features: ``x = (x − μ) / σ`` per column.
    4. **Squeeze** ``y`` to 1-D if the saved tensor is 2-D.
    5. **Select fold** when masks have shape ``[N, n_folds]``.
    6. **Binarise** labels: 1 = fraud (class 1), 0 = everything else.
    7. **Materialise** ``edge_index`` from ``adj_t`` for NeighborLoader.

    Parameters
    ----------
    data_dir:
        Parent datasets directory.  Must contain ``DGraphFin/dgraphfin.npz``.
        The processed cache is written to ``processed_data/`` at the project
        root (one level above *data_dir*).
    fold:
        Which column to select when the dataset ships multiple splits.
        Ignored (silently) when masks are 1-D.

    Returns
    -------
    :class:`DGraphFinBundle`
    """
    # ── Step 1: load (pre_transform builds & caches adj_t) ──────────────────
    # root = datasets/DGraphFin  →  raw_dir       = datasets/DGraphFin/
    #                               processed_dir = processed_data/
    dataset = DGraphFin(
        root          = os.path.join(data_dir, 'DGraphFin'),
        pre_transform = T.ToSparseTensor(),
    )
    data = dataset[0]

    # ── Step 2: symmetrise adjacency ────────────────────────────────────────
    # Mirrors: data.adj_t = data.adj_t.to_symmetric()
    # Ensures every directed edge u→v also exists as v→u for aggregation.
    data.adj_t = data.adj_t.to_symmetric()

    # ── Step 3: z-score feature normalisation ───────────────────────────────
    # Mirrors: x = (x - x.mean(0)) / x.std(0)
    x      = data.x
    x      = (x - x.mean(0)) / x.std(0)
    data.x = x

    # ── Step 4: squeeze label tensor ────────────────────────────────────────
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    node_feat_dim = data.num_features

    # ── Step 5: fold-aware split selection ──────────────────────────────────
    # DGraphFin ships a single split, so masks are 1-D index tensors.
    # Guard for any future multi-fold variant.
    train_idx = data.train_mask
    val_idx   = data.valid_mask
    test_idx  = data.test_mask

    if train_idx.dim() > 1 and train_idx.shape[1] > 1:
        train_idx = train_idx[:, fold]
        val_idx   = val_idx[:, fold]
        test_idx  = test_idx[:, fold]

    # ── Step 6: binary labels ────────────────────────────────────────────────
    # The masks cover class-0 (normal) and class-1 (fraud) nodes only.
    # Class-2 and class-3 background nodes are in the graph for neighbourhood
    # aggregation but never appear in any split index tensor.
    # Binary target: 1 = fraud, 0 = normal.
    y_binary      = (data.y == 1).float()
    train_labels  = y_binary[train_idx].numpy()

    # ── Step 7: materialise symmetric edge_index for NeighborLoader ─────────
    # T.ToSparseTensor() stores connectivity in data.adj_t (transposed CSR).
    # After .to_symmetric() it encodes both directions.  We recover COO
    # edge_index because NeighborLoader requires it.
    row, col, _    = data.adj_t.t().coo()
    edge_index_sym = torch.stack([row, col], dim=0)

    # Assemble the graph object that NeighborLoader will sample from.
    # y_binary is attached so batch.y[:batch_size] gives float targets directly.
    graph = Data(
        x          = data.x,
        edge_index = edge_index_sym,
        y          = y_binary,
        num_nodes  = data.num_nodes,
    )

    return DGraphFinBundle(
        graph         = graph,
        train_idx     = train_idx,
        val_idx       = val_idx,
        test_idx      = test_idx,
        train_labels  = train_labels,
        node_feat_dim = node_feat_dim,
    )


def load_dgraphfin_temporal(data_dir, fold=0, max_time_steps=32):
    dataset = DGraphFin(root=os.path.join(data_dir, 'DGraphFin'), pre_transform=T.ToSparseTensor())
    data    = dataset[0]

    # z-score normalise features
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    node_feat_dim = data.num_features
    N             = data.num_nodes

    # directed edge_index from adj_t  (row=dst, col=src)
    row_t, col_t, _ = data.adj_t.coo()
    src = col_t
    dst = row_t
    edge_index_directed = torch.stack([src, dst], dim=0)

    # edge_time: shift → normalise → discretise → reshape  (process_data steps 1-4)
    et = data.edge_time.float()
    et = et - et.min()
    et = et / et.max()
    et = (et * max_time_steps).long()
    et = et.view(-1, 1).float()

    # node_time = min out-edge time per node  (groupby-min)
    edge_np     = torch.cat([edge_index_directed, et.view(1, -1)], dim=0).T.numpy()
    df          = pd.DataFrame(edge_np, columns=['src', 'dst', 'time'])
    min_time_df = df.groupby('src')['time'].min()
    node_time_np = np.zeros(N, dtype=np.float32)
    node_time_np[min_time_df.index.astype(int)] = min_time_df.values.astype(np.float32)
    node_time = torch.tensor(node_time_np)

    # symmetrise: repeat same edge_time for both directions
    edge_index_sym = torch.cat([edge_index_directed, edge_index_directed[[1, 0], :]], dim=1)
    edge_attr_sym  = torch.cat([et, et], dim=0)

    # split masks
    train_idx = data.train_mask
    val_idx   = data.valid_mask
    test_idx  = data.test_mask
    if train_idx.dim() > 1 and train_idx.shape[1] > 1:
        train_idx = train_idx[:, fold]
        val_idx   = val_idx[:, fold]
        test_idx  = test_idx[:, fold]

    y_binary     = (data.y == 1).float()
    train_labels = y_binary[train_idx].numpy()

    graph = Data(
        x          = data.x,
        edge_index = edge_index_sym,
        edge_attr  = edge_attr_sym,
        node_time  = node_time,
        y          = y_binary,
        num_nodes  = N,
    )
    return graph, train_idx, val_idx, test_idx, train_labels, node_feat_dim