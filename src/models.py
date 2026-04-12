import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, TransformerConv, FAConv
from torch_geometric.nn import MessagePassing
import numpy as np

################## model definitions ##################
# In NeighborLoader mini-batch training the subgraph for each batch contains
# seed nodes + their sampled neighbours. Seed nodes are always placed first
# (indices 0..batch_size-1). The full GNN runs on the subgraph; only seed
# nodes are classified.

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, dropout):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch_size):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.clf(h[:batch_size]).squeeze(-1)


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, heads, dropout):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATConv(in_channels, hidden_channels // heads,
                                   heads=heads, dropout=dropout, concat=True))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                       heads=heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch_size):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.clf(h[:batch_size]).squeeze(-1)


class GATv2Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, heads, dropout):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATv2Conv(in_channels, hidden_channels // heads,
                                     heads=heads, dropout=dropout, concat=True))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels // heads,
                                         heads=heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch_size):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.clf(h[:batch_size]).squeeze(-1)
    

class FAGCNModel(torch.nn.Module):
    """
    Frequency Adaptive Graph Convolution Network (FAGCN).
    Bo et al., "Beyond Low-Frequency Information in Graph Convolutional Networks"
    AAAI 2021.
 
    FAConv computes signed, adaptive attention weights α ∈ (-1, 1) via tanh,
    allowing it to capture both low-frequency (homophilic) and high-frequency
    (heterophilic) signals depending on the graph structure.
 
    Update rule per layer:
        x'_i = ε · x_0_i  +  Σ_{j∈N(i)}  α_ij / √(d_i·d_j)  · x_j
    where x_0 is the initial feature (residual connection to input),
    ε is a learnable scalar, and α_ij = tanh(a^T [x_i || x_j]).
 
    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    n_layers : int
    dropout : float
    eps : float
        Initial value of the ε residual weight (default 0.1).
    """
 
    def __init__(self, in_channels, hidden_channels, n_layers, dropout, eps=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs   = torch.nn.ModuleList()
        self.norms   = torch.nn.ModuleList()
 
        # FAConv requires same in/out channels (it only aggregates, no projection).
        # Use an input linear layer to project in_channels → hidden_channels first,
        # then all FAConv layers operate at hidden_channels.
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
 
        for _ in range(n_layers):
            # channels = hidden_channels (both input and output are same dim)
            # cached=False: required for mini-batch / inductive setting
            self.convs.append(FAConv(hidden_channels, eps=eps, dropout=dropout,
                                     cached=False, add_self_loops=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
 
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )
 
    def forward(self, x, edge_index, batch_size):
        # Project to hidden dim and store as x_0 (initial residual for all layers)
        x_0 = F.relu(self.input_proj(x))     # [N, hidden]
        h   = x_0
 
        for conv, norm in zip(self.convs, self.norms):
            # FAConv.forward(x, x_0, edge_index)
            # x   : current node embeddings
            # x_0 : initial node embeddings (residual anchor, fixed across layers)
            h = norm(conv(h, x_0, edge_index).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)
 
        return self.clf(h[:batch_size]).squeeze(-1)
    

# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time encoding  (Bochner / TGAT paper eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

class TimeEncode(torch.nn.Module):
    """
    Learnable sinusoidal time encoding from the original TGAT implementation:
    https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

    Maps a scalar time value t to a time_dim-dimensional vector:

        φ(t) = cos(t · basis_freq + phase)

    where basis_freq and phase are **learned** parameters initialised as:
        basis_freq  ~ log-spaced  1 / 10^linspace(0, 9, time_dim)
        phase       ~ zeros

    Parameters
    ----------
    expand_dim : int
        Output dimensionality (called time_dim elsewhere in this file).
    factor : int
        Stored for compatibility with the original API; not used in forward.
    """

    def __init__(self, expand_dim: int, factor: int = 5):
        super().__init__()
        time_dim    = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim)
            ).float()
        )                                             # [time_dim]  learned
        self.phase = torch.nn.Parameter(
            torch.zeros(time_dim).float()
        )                                             # [time_dim]  learned

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts : Tensor  shape [N, L]
            Raw time scalars arranged as a 2-D batch
            (N = nodes/edges, L = sequence length; L=1 for single timestamps).

        Returns
        -------
        Tensor  shape [N, L, time_dim]
        """
        batch_size = ts.size(0)
        seq_len    = ts.size(1)

        ts      = ts.view(batch_size, seq_len, 1)              # [N, L, 1]
        map_ts  = ts * self.basis_freq.view(1, 1, -1)          # [N, L, time_dim]
        map_ts  = map_ts + self.phase.view(1, 1, -1)           # [N, L, time_dim]

        harmonic = torch.cos(map_ts)                           # [N, L, time_dim]
        return harmonic


# ─────────────────────────────────────────────────────────────────────────────
# TGAT with TGL Parallel Sampler
# ─────────────────────────────────────────────────────────────────────────────
class TGATSamplerModel(torch.nn.Module):
    """
    TGAT variant that accepts pre-sampled batch_inputs from the TGL C++
    ParallelSampler via to_dgl_blocks + to_pyg_inputs.
 
    Compared to TGAT:
    - dts (Δt per edge) is provided directly from the sampler — no need to
      recompute rel_t inside the model.
    - node features are looked up from the global x_all matrix using node_ids
      returned by the sampler, so x is not passed directly.
    - Works with HISTORY=1 (TGAT-style, one temporal snapshot per layer).
 
    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    n_smp_layers : int
        Number of SMP layers (L in paper). 0 = TMP only.
    time_dim : int
    dropout : float
    """
 
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_layers:        int,
        n_head:          int,
        time_dim:        int,
        dropout:         float = 0.2,
        feature_augment: bool = False,
    ):
        super().__init__()
        assert hidden_channels % n_head == 0, \
            f'hidden_channels ({hidden_channels}) must be divisible by n_head ({n_head})'

        self.time_enc   = TimeEncode(time_dim)
        self.feature_augment = feature_augment
        if self.feature_augment:
            in_channels += 7

        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.dropout    = dropout
        self.n_layers   = n_layers

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(
                TransformerConv(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels // n_head,
                    heads        = n_head,
                    edge_dim     = time_dim,
                    dropout      = dropout,
                    concat       = True,
                    beta         = False,
                )
            )
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def _augment_features(
        self,
        x_base:     torch.Tensor,   # [N_sub, base_in_channels]
        # node_ids:   torch.Tensor,   # [N_sub]  global ids (unused here, kept for API)
        edge_index: torch.Tensor,   # [2, E]   local index space
        edge_dts:   torch.Tensor,   # [E]      Δt = query_ts[dst] - edge_ts
        n_seeds:    int,
    ) -> torch.Tensor:              # [N_sub, base_in_channels + N_EXTRA]
        """
        Compute structural and temporal features from the sampled subgraph
        and concatenate with base node features.
 
        All features are derived solely from edges the TGL sampler returned
        (t_edge < t_query enforced by sampler) — no additional leakage risk.
        Normalised using seed-node statistics only to avoid neighbour
        distribution leakage.
        """
        n_sub  = x_base.shape[0]
        device = x_base.device
        src_l, dst_l = edge_index                        # [E] local indices
 
        ones = torch.ones(src_l.shape[0], device=device)
 
        # ── structural ────────────────────────────────────────────────────────
        out_deg   = torch.zeros(n_sub, device=device).scatter_add_(0, src_l, ones)
        in_deg    = torch.zeros(n_sub, device=device).scatter_add_(0, dst_l, ones)
        deg_ratio = out_deg / (in_deg + 1.0)             # avoids div-by-zero
 
        # ── temporal (edge_dts = query_ts[dst] - edge_ts, larger = older) ────
        INF = torch.full((n_sub,), 1e9, device=device)
 
        # most recent edge per dst: smallest dts value
        min_dts = INF.clone().scatter_reduce_(
            0, dst_l, edge_dts, reduce='amin', include_self=True
        )
        # oldest edge per dst: largest dts value
        max_dts = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts, reduce='amax', include_self=True
        )
        # recency: how recent is the node's newest valid edge
        recency = min_dts.clamp(max=1e8)                 # cap sentinels
 
        # burst ratio: fraction of edges in the most recent 25% of activity window
        activity_window = (max_dts - min_dts).clamp(min=1.0)
        burst_cutoff    = min_dts + 0.25 * activity_window   # [N_sub]
        is_burst        = (edge_dts <= burst_cutoff[dst_l]).float()
        burst_count     = torch.zeros(n_sub, device=device).scatter_add_(
            0, dst_l, is_burst
        )
        burst_ratio = burst_count / in_deg.clamp(min=1.0)
 
        # mean and std of dts per dst node (temporal diversity of neighbourhood)
        mean_dts = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts, reduce='mean', include_self=False
        )
        dts_sq   = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts ** 2, reduce='mean', include_self=False
        )
        std_dts  = (dts_sq - mean_dts ** 2).clamp(min=0).sqrt()
 
        # ── stack extra features ──────────────────────────────────────────────
        extra = torch.stack([
            out_deg, in_deg, deg_ratio,
            recency, burst_ratio, mean_dts, std_dts,
        ], dim=1).float()                                # [N_sub, N_EXTRA]
 
        # normalise using seed-node mean/std to avoid neighbour dist leakage
        mean = extra[:n_seeds].mean(0)
        std  = extra[:n_seeds].std(0).clamp(min=1e-8)
        extra = (extra - mean) / std
 
        return torch.cat([x_base, extra], dim=1)         # [N_sub, base+N_EXTRA]
 
    def forward(
        self,
        x_all:       torch.Tensor,   # [N_total, in_channels]  full node feature matrix on device
        batch_inputs: list,           # output of to_pyg_inputs(mfgs)
        batch_size:  int,             # number of seed nodes
    ) -> torch.Tensor:               # [batch_size] logits
 
        # batch_inputs[layer][hist=0] = dict with node_ids, edge_index, edge_dts
        # innermost layer = batch_inputs[-1][0]
        inner = batch_inputs[-1][0]
 
        node_ids   = inner['node_ids']    # [N_sub]  global node ids
        edge_index = inner['edge_index']  # [2, E]   local index space
        dts        = inner['edge_dts']    # [E]      Δt = query_ts[dst] - edge_ts
 
        # Look up node features from global matrix
        x = x_all[node_ids]              # [N_sub, in_channels]
        # feature augment
        if self.feature_augment:
            x      = self._augment_features(
                x, edge_index, dts, batch_size
            )
 
        # ── time encoding ─────────────────────────────────────────────────────
        # dts is already Δt per edge from the TGL sampler — feed directly to
        # TimeEncode. Unsqueeze to [E, 1] for TimeEncode's [N, L] input format.
        rel_t_enc = self.time_enc(dts.float().unsqueeze(1)).squeeze(1)  # [E, time_dim]
 
        # ── input projection ──────────────────────────────────────────────────
        h = F.relu(self.input_proj(x))                            # [N, hidden]

        # ── conv layers ───────────────────────────────────────────────────────
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index, edge_attr=rel_t_enc).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)

        # ── classify seed nodes only ──────────────────────────────────────────
        return self.clf(h[:batch_size]).squeeze(-1)               # [batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# THEGCN model
# ─────────────────────────────────────────────────────────────────────────────
class TMPConv(MessagePassing):
    """
    Temporal Message Passing block for THEGCN.
    
    For each directed edge (src → dst) with relative time Δt = t\'_dst - t_edge:
        p = σ( MLP([x_src || x_dst || E(Δt)]) )        low-pass weight ∈ (0,1)
        q = 1 - p                                        high-pass weight
        message = (p - q) * x_src                       signed, can be negative
 
    Aggregation (Eq. 5):
        h_dst^(1) = x_dst + mean_{neighbours} (p - q) * x_src
 
    Uses mean aggregation (1/|S| factor from paper). Note: the paper says
    message = (p - q) * x_src where x_src is the *source node feature at time t*.
    In our mini-batch setting we use the static node feature x_src as an
    approximation since DGraphFin has time-invariant node features.
    """
    def __init__(self, in_channels: int, time_dim: int):
        super().__init__(aggr='mean')
        
        # MLP to compute low-pass weight p from [x_src || x_dst || E(Δt)]
        mlp_in = 2 * in_channels + time_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, mlp_in//2),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_in//2, in_channels),
        )

    def forward(
        self,
        x: torch.Tensor,            # [N, in_channels]: node features of target node at time t'
        edge_index: torch.Tensor,   # [2, E]: sampled interaction edges (src)
        rel_t_enc: torch.Tensor,    # [E, time_dim]
    ) -> torch.Tensor:              # [N, in_channels]
        # propagate calls message() then aggregate with mean
        agg = self.propagate(edge_index, x=x, rel_t_enc=rel_t_enc)
        return x + agg # residual connection
    
    def message(
            self,
            x_i: torch.Tensor,      # [E, in_channels]: source node features
            x_j: torch.Tensor,      # [E, in_channels]: destination node features
            rel_t_enc: torch.Tensor # [E, time_dim]
            ) -> torch.Tensor:          # [E, in_channels]
        # compute low pass weight p
        mlp_in = torch.cat([x_i, x_j, rel_t_enc], dim=-1)   # [E, 2*in_channels + time_dim]
        p = torch.tanh(self.mlp(mlp_in))                    # or sigmoid
        # high pass weight q=1-p
        q = 1.0 - p
        return (p-q) * x_i  # signed message


class SMPConv(MessagePassing):
    """
    Static Messsage Passing block for THEGCN.
    Operates on node embeddings after the TMP block. No time information.
        p = σ( MLP([h_src^(l) || h_dst^(l)]) )         low-pass weight ∈ (0,1)
        q = 1 - p
        h_dst^(l+1) = h_dst^(l) + mean_{neighbours} (p - q) * h_src^(l)
 
    The paper uses tanh in the ablation to allow negative weights, but the
    main model uses sigmoid (non-negative p, q but p-q can still be negative
    when p < 0.5). We follow the main model formulation.
    """
    def __init__(self, hidden_channels: int):
        super().__init__(aggr='mean')
        mlp_in = hidden_channels * 2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, mlp_in//2),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_in//2, hidden_channels),
        )

    def forward(
            self,
            h: torch.Tensor,            # [N, hidden_channels]: node embeddings after TMP
            edge_index: torch.Tensor,   # [2, E]: sampled interaction edges (src)
    ) -> torch.Tensor:              # [N, hidden_channels]
        agg = self.propagate(edge_index, h=h)
        return h + agg # residual connection
    
    def message(
            self,
            h_i: torch.Tensor,      # [E, hidden_channels]: source node embeddings
            h_j: torch.Tensor       # [E, hidden_channels]: destination node embeddings
    ) -> torch.Tensor:          # [E, hidden_channels]
        mlp_in = torch.cat([h_i, h_j], dim=-1)   # [E, 2*hidden_channels]
        p = torch.tanh(self.mlp(mlp_in))         # or sigmoid
        q = 1.0 - p
        return (p-q) * h_i  # signed message
    

# ─────────────────────────────────────────────────────────────────────────────
# THEGCN with TGL Parallel Sampler
# ─────────────────────────────────────────────────────────────────────────────
class THEGCNSamplerModel(torch.nn.Module):
    """
    THEGCN variant that accepts pre-sampled batch_inputs from the TGL C++
    ParallelSampler via to_dgl_blocks + to_pyg_inputs.
 
    Compared to THEGCNModel:
    - dts (Δt per edge) is provided directly from the sampler — no need to
      recompute rel_t inside the model.
    - node features are looked up from the global x_all matrix using node_ids
      returned by the sampler, so x is not passed directly.
    - Works with HISTORY=1 (TGAT-style, one temporal snapshot per layer).
 
    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    n_smp_layers : int
        Number of SMP layers (L in paper). 0 = TMP only.
    time_dim : int
    dropout : float
    """
 
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_smp_layers:    int,
        time_dim:        int,
        dropout:         float = 0.2,
        feature_augment: bool = False,
    ):
        super().__init__()
        self.time_enc     = TimeEncode(time_dim)
        self.dropout      = dropout
        self.n_smp_layers = n_smp_layers
        self.feature_augment = feature_augment
        
        if self.feature_augment:
            in_channels += 7
 
        # TMP block: operates on raw node features
        self.tmp = TMPConv(in_channels=in_channels, time_dim=time_dim)
 
        # Linear projection: in_channels → hidden_channels before SMP
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
 
        # SMP block: L layers of static heterophily-aware message passing
        self.smp_convs = torch.nn.ModuleList([
            SMPConv(hidden_channels) for _ in range(n_smp_layers)
        ])
        self.smp_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_channels) for _ in range(n_smp_layers)
        ])
 
        # Classifier MLP — same structure as static GNN baselines
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )

    def _augment_features(
        self,
        x_base:     torch.Tensor,   # [N_sub, base_in_channels]
        # node_ids:   torch.Tensor,   # [N_sub]  global ids (unused here, kept for API)
        edge_index: torch.Tensor,   # [2, E]   local index space
        edge_dts:   torch.Tensor,   # [E]      Δt = query_ts[dst] - edge_ts
        n_seeds:    int,
    ) -> torch.Tensor:              # [N_sub, base_in_channels + N_EXTRA]
        """
        Compute structural and temporal features from the sampled subgraph
        and concatenate with base node features.
 
        All features are derived solely from edges the TGL sampler returned
        (t_edge < t_query enforced by sampler) — no additional leakage risk.
        Normalised using seed-node statistics only to avoid neighbour
        distribution leakage.
        """
        n_sub  = x_base.shape[0]
        device = x_base.device
        src_l, dst_l = edge_index                        # [E] local indices
 
        ones = torch.ones(src_l.shape[0], device=device)
 
        # ── structural ────────────────────────────────────────────────────────
        out_deg   = torch.zeros(n_sub, device=device).scatter_add_(0, src_l, ones)
        in_deg    = torch.zeros(n_sub, device=device).scatter_add_(0, dst_l, ones)
        deg_ratio = out_deg / (in_deg + 1.0)             # avoids div-by-zero
 
        # ── temporal (edge_dts = query_ts[dst] - edge_ts, larger = older) ────
        INF = torch.full((n_sub,), 1e9, device=device)
 
        # most recent edge per dst: smallest dts value
        min_dts = INF.clone().scatter_reduce_(
            0, dst_l, edge_dts, reduce='amin', include_self=True
        )
        # oldest edge per dst: largest dts value
        max_dts = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts, reduce='amax', include_self=True
        )
        # recency: how recent is the node's newest valid edge
        recency = min_dts.clamp(max=1e8)                 # cap sentinels
 
        # burst ratio: fraction of edges in the most recent 25% of activity window
        activity_window = (max_dts - min_dts).clamp(min=1.0)
        burst_cutoff    = min_dts + 0.25 * activity_window   # [N_sub]
        is_burst        = (edge_dts <= burst_cutoff[dst_l]).float()
        burst_count     = torch.zeros(n_sub, device=device).scatter_add_(
            0, dst_l, is_burst
        )
        burst_ratio = burst_count / in_deg.clamp(min=1.0)
 
        # mean and std of dts per dst node (temporal diversity of neighbourhood)
        mean_dts = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts, reduce='mean', include_self=False
        )
        dts_sq   = torch.zeros(n_sub, device=device).scatter_reduce_(
            0, dst_l, edge_dts ** 2, reduce='mean', include_self=False
        )
        std_dts  = (dts_sq - mean_dts ** 2).clamp(min=0).sqrt()
 
        # ── stack extra features ──────────────────────────────────────────────
        extra = torch.stack([
            out_deg, in_deg, deg_ratio,
            recency, burst_ratio, mean_dts, std_dts,
        ], dim=1).float()                                # [N_sub, N_EXTRA]
 
        # normalise using seed-node mean/std to avoid neighbour dist leakage
        mean = extra[:n_seeds].mean(0)
        std  = extra[:n_seeds].std(0).clamp(min=1e-8)
        extra = (extra - mean) / std
 
        return torch.cat([x_base, extra], dim=1)         # [N_sub, base+N_EXTRA]
 
    def forward(
        self,
        x_all:       torch.Tensor,   # [N_total, in_channels]  full node feature matrix on device
        batch_inputs: list,           # output of to_pyg_inputs(mfgs)
        batch_size:  int,             # number of seed nodes
    ) -> torch.Tensor:               # [batch_size] logits
 
        # batch_inputs[layer][hist=0] = dict with node_ids, edge_index, edge_dts
        # innermost layer = batch_inputs[-1][0]
        inner = batch_inputs[-1][0]
        node_ids   = inner['node_ids']    # [N_sub]  global node ids
        edge_index = inner['edge_index']  # [2, E]   local index space
        dts        = inner['edge_dts']    # [E]      Δt = query_ts[dst] - edge_ts
 
        # Look up node features from global matrix
        x = x_all[node_ids]              # [N_sub, in_channels]
        # feature augment
        if self.feature_augment:
            x      = self._augment_features(
                x, edge_index, dts, batch_size
            )                                                                # [N_sub, base+N_EXTRA]
 
        # ── time encoding ─────────────────────────────────────────────────────
        # dts is already Δt per edge from the TGL sampler — feed directly to
        # TimeEncode. Unsqueeze to [E, 1] for TimeEncode's [N, L] input format.
        rel_t_enc = self.time_enc(dts.float().unsqueeze(1)).squeeze(1)  # [E, time_dim]
 
        # ── TMP block (Eq. 4-5) ──────────────────────────────────────────────
        h = self.tmp(x, edge_index, rel_t_enc)                          # [N_sub, in_channels]
        h = F.dropout(h, p=self.dropout, training=self.training)
 
        # ── project to hidden dim ─────────────────────────────────────────────
        h = F.relu(self.input_proj(h))                                  # [N_sub, hidden]
 
        # ── SMP block (Eq. 6-7) ──────────────────────────────────────────────
        # Reuses the same edge_index (topology fixed after TMP).
        # For multi-hop models, outer layers could use batch_inputs[:-1]
        # but for simplicity and following the paper we use the same subgraph.
        for conv, norm in zip(self.smp_convs, self.smp_norms):
            h = norm(F.relu(conv(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
 
        # ── classify seed nodes only ──────────────────────────────────────────
        # Seed nodes are dst nodes in the DGL block, which to_pyg_inputs places
        # at positions 0..batch_size-1 in the destination ordering.
        # DGL block: dst nodes come first in srcdata['ID'], so h[:batch_size]
        # corresponds to the seed nodes.
        return self.clf(h[:batch_size]).squeeze(-1)                     # [batch_size]
 