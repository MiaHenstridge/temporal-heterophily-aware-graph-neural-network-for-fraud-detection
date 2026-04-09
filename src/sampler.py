"""
tgl_sampler.py
==============
Python wrapper around the compiled C++ ParallelSampler (sampler_core) from TGL.
 
Usage
-----
Build the C++ extension first:
    cd ~/notebooks/final_thesis
    pip install pybind11
    python src/setup.py build_ext --inplace
 
Then in your training script:
 
    from tgl_sampler import TemporalSampler, load_sampler_data
 
    # load the pre-built CSC graph (built once by prepare_sampler_data.py)
    graph_data = load_sampler_data('./processed_data/tgl/dgraphfin_tgl.npz')
 
    sampler = TemporalSampler(
        graph_data      = graph_data,
        num_neighbors   = [10, 5],      # per layer, outer → inner
        num_workers     = 1,
        num_threads     = 8,
        recent          = True,         # pick most-recent N neighbours (TGAT style)
    )
 
    # at training time (one batch of seed nodes)
    blocks = sampler.sample(seed_nodes, seed_timestamps)
    # blocks[0] = outermost layer subgraph, blocks[-1] = innermost (seed layer)
 
Data format expected in graph_data (numpy .npz):
    indptr   int32   [N+1]   CSC column pointers
    indices  int32   [E]     CSC row indices = source node ids
    eid      int32   [E]     original edge ids (0..E-1)
    ts       float32 [E]     edge timestamps
    num_nodes int
    num_edges int
 
    Within each node's slice [indptr[n] : indptr[n+1]], edges must be sorted
    by timestamp (ascending). prepare_sampler_data.py guarantees this.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import List


try:
    import sampler_core
except ImportError:
    raise ImportError(
        "sampler_core not found. Build it with:\n"
        "  python src/setup.py build_ext --inplace\n"
        "from the project root."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data container returned by load_sampler_data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TGLGraphData:
    """
    CSC-format temporal graph ready for ParallelSampler.
    """
    indptr:     np.ndarray    # int32 [N+1]
    indices:    np.ndarray    # int32 [E] source node ids
    eid:        np.ndarray    # int32 [E] original edge ids
    ts:         np.ndarray    # float32 [E] edge timestamps
    num_nodes:  int 
    num_edges:  int


def load_sampler_data(path: str) -> TGLGraphData:
    """Load pre-built CSC graph from a .npz file"""
    data = np.load(path)
    return TGLGraphData(
        indptr      = data['indptr'].astype(np.int32),
        indices     = data['indicies'].astype(np.int32),
        eid         = data['eid'].astype(np.int32),
        ts          = data['ts'].astype(np.float32),
        num_nodes   = int(data['num_nodes']),
        num_edges   = int(data['num_edges'])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampled block — one hop's subgraph in PyG-friendly form
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SampledBlock:
    """
    One layer's sampled subgraph, converted from TemporalGraphBlock

    Attributes
    ----------
    nodes       : LongTensor  [dim_in]   global node ids (seeds first)
    edge_index  : LongTensor  [2, E]     directed edges in LOCAL index space
                                          row=0 → src (neighbour), row=1 → dst (seed)
    eid         : LongTensor  [E]        original edge ids
    ts          : FloatTensor [E]        edge timestamps
    dts         : FloatTensor [E]        Δt = query_ts[dst] - edge_ts  (time encoding input)
    seed_ts     : FloatTensor [dim_out]  query timestamps for seed nodes
    n_seeds     : int                    number of seed nodes (== dim_out)
    n_total     : int                    total nodes in block (== dim_in)
    """
    nodes:      torch.Tensor
    edge_index: torch.Tensor
    eid:        torch.Tensor
    ts:         torch.Tensor
    dts:        torch.Tensor
    seed_ts:    torch.Tensor
    n_seeds:    int
    n_total:    int

    def to(self, device):
        self.nodes      = self.nodes.to(device)
        self.edge_index = self.edge_index.to(device)
        self.eid        = self.eid.to(device)
        self.ts         = self.ts.to(device)
        self.dts        = self.dts.to(device)
        self.seed_ts    = self.seed_ts.to(device)
        return self
    

def _block_from_tgb(tgb) -> SampledBlock:
    """Convert a C++ TemporalGraphBlock to a SampledBlock"""
    nodes = torch.from_numpy(tgb.nodes().astype(np.int64))      # [dim_in]
    row = torch.from_numpy(tgb.col().astype(np.int64))          # [E] src local idx
    col = torch.from_numpy(tgb.row().astype(np.int64))          # [E] dst local idx

    # NOTE: C++ block stores row = dst-side (root), col = src-side (neighbour).
    # We swap to standard PyG convention: edge_index[0]=src, edge_index[1]=dst.
    edge_index = torch.stack([row, col], dim=0)               # [2, E]
 
    eid  = torch.from_numpy(tgb.eid().astype(np.int64))      # [E]

    # ts and dts are laid out as [seed_ts..., edge_ts...]
    n_seeds = tgb.dim_out
    full_ts = tgb.ts()
    full_dts = tgb.dts()

    seed_ts = torch.from_numpy(full_ts[:n_seeds].astype(np.float32))
    edge_ts = torch.from_numpy(full_ts[n_seeds:].astype(np.float32))
    edge_dts = torch.from_numpy(full_dts[n_seeds:].astype(np.float32))

    return SampledBlock(
        nodes = nodes,
        edge_index=edge_index,
        eid=eid,
        ts=edge_ts,
        dts=edge_dts,
        seed_ts=n_seeds,
        n_total=tgb.dim_in,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main sampler class
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSampler:
    """
    Python wrapper around the C++ ParallelSampler.
 
    Produces temporally-constrained neighbourhood subgraphs for THEGCN/TGAT
    mini-batch training.  Each call to sample() returns one SampledBlock per
    GNN layer (outermost hop first, innermost last).
 
    Parameters
    ----------
    graph_data : TGLGraphData
        Pre-built CSC graph from load_sampler_data().
    num_neighbors : List[int]
        Number of neighbours to sample per layer.
        len(num_neighbors) == number of GNN layers.
    num_workers : int
        Number of parallel workers (default 1; set to number of CPU sockets).
    num_threads : int
        Total OpenMP threads (num_thread_per_worker = num_threads // num_workers).
    recent : bool
        True  → pick the most recent N neighbours (TGAT-style, recommended).
        False → sample uniformly at random within the valid time window.
    prop_time : bool
        True  → propagate the root query timestamp to sampled neighbours.
        False → use the actual edge timestamp (recommended for THEGCN).
    """
    def __init__(
        self,
        graph_data: TGLGraphData,
        num_neighbors: List[int],
        num_workers: int=4,
        num_threads: int=8,
        recent: bool=True,
        prop_time: bool=False
    ):
        self.graph_data = graph_data
        self.num_neighbors = num_neighbors
        self.num_layers = len(num_neighbors)
        self.num_workers = num_workers
        self.num_threads = num_threads

        threads_per_worker = max(1, num_threads // num_workers)

        # build the C++ sampler
        # num_history=1, window_duration=0 -> TGAT-style (sample all events before t)
        self._sampler = sampler_core.ParallelSampler(
            graph_data.indptr.tolist(),
            graph_data.indices.tolist(),
            graph_data.eid.tolist(),
            graph_data.ts.tolist(),
            threads_per_worker,
            num_workers,
            self.num_layers,
            num_neighbors,
            recent,
            prop_time,
            1,                              # num_history (1=TGAT-style)
            0.0,                            # window_duration (unseen when num_history=1)
        )

    
    def reset(self):
        """Reset internal timestamp pointers (call between epochs if needed)"""
        self._sampler.reset()

    def sample(
        self,
        seed_nodes: torch.Tensor,   # [B] LongTensor of seed node ids
        seed_ts: torch.Tensor,      # [B] FloatTensor of query timestamps  
    ) -> List[SampledBlock]:
        """
        Sample temporal neighborhoods for a batch of seed nodes.

        Parameters
        ----------
        seed_nodes : LongTensor [B]
        seed_ts    : FloatTensor [B]
 
        Returns
        -------
        List[SampledBlock]
            Length == num_layers.
            blocks[0] = outermost hop (2nd-order neighbours for a 2-layer model).
            blocks[-1] = innermost hop (direct neighbours of seeds).
 
            For THEGCN forward pass use blocks[-1] (the direct-neighbour block)
            for TMP and subsequently pass blocks[0...-1] to the SMP layers.
        """
        root_nodes = seed_nodes.cpu().numpy().astype(np.int32).tolist()
        root_ts    = seed_ts.cpu().numpy().astype(np.float32).tolist()
 
        self._sampler.sample(root_nodes, root_ts)
        raw_blocks = self._sampler.get_ret()
 
        # raw_blocks has length num_layers * num_history (= num_layers for history=1)
        # Order: outermost hop first.
        return [_block_from_tgb(b) for b in raw_blocks]

    def __repr__(self):
        return (
            f"TemporalSampler("
            f"nodes={self.graph_data.num_nodes:,}, "
            f"edges={self.graph_data.num_edges:,}, "
            f"layers={self.num_layers}, "
            f"neighbors={self.num_neighbors}, "
            f"threads={self.num_threads})"
        )