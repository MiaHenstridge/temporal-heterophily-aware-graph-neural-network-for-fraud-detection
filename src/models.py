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
# TGAT model
# ─────────────────────────────────────────────────────────────────────────────
class TGATModel(torch.nn.Module):
    """
    Temporal Graph Attention Network for node classification.

    Parameters
    ----------
    in_channels : int
        Raw node feature dimension (17 for DGraphFin).
    hidden_channels : int
        Hidden dimension (must be divisible by n_head).
    n_layers : int
        Number of TGAT layers to stack.
    n_head : int
        Number of attention heads in TransformerConv.
    time_dim : int
        Dimension of the sinusoidal time encoding.
    dropout : float
        Dropout probability applied after each conv layer and in the MLP.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_layers:        int,
        n_head:          int,
        time_dim:        int,
        dropout:         float = 0.2,
    ):
        super().__init__()
        assert hidden_channels % n_head == 0, \
            f'hidden_channels ({hidden_channels}) must be divisible by n_head ({n_head})'

        self.time_enc   = TimeEncode(time_dim)
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

    def forward(
        self,
        x:          torch.Tensor,   # [N_sub, in_channels]
        edge_index: torch.Tensor,   # [2, E_sub]
        time:       torch.Tensor,   # [E_sub, 1]   raw edge timestamps  (graph.time)
        node_time:  torch.Tensor,   # [N_sub]      node timestamps       (graph.node_time)
        batch_size: int,
    ) -> torch.Tensor:              # [batch_size]  logits

        # ── relative time per edge ────────────────────────────────────────────
        # time is [E] (1-D) as required by PyG's temporal NeighborLoader.
        # Unsqueeze to [E, 1] for the subtraction and TimeEncode.
        # rel_t = node_time[dst] - edge_time  →  [E, 1]
        rel_t     = node_time[edge_index[1]] - time               # [E]
        rel_t_enc = self.time_enc(rel_t.float().unsqueeze(1)).squeeze(1)  # [E, time_dim]

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
        p = torch.sigmoid(self.mlp(mlp_in))
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
        p = torch.sigmoid(self.mlp(mlp_in))
        q = 1.0 - p
        return (p-q) * h_i  # signed message
    
    


class THEGCNModel(torch.nn.Module):
    """
    Temporal Heterophilic Graph Convolutional Network (THEGCN).
    Yan et al., 2025  (arXiv:2412.16435)
 
    Parameters
    ----------
    in_channels : int
        Raw node feature dimension (17 for DGraphFin).
    hidden_channels : int
        Hidden dimension for the SMP layers and classifier.
    n_smp_layers : int
        Number of SMP layers (L in the paper). 0 = TMP only.
    time_dim : int
        Dimension of the sinusoidal time encoding E(Δt).
    dropout : float
        Dropout applied after TMP and each SMP layer, and in the classifier.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_smp_layers:    int,
        time_dim:        int,
        dropout:         float = 0.2,
    ):
        super().__init__()

        self.time_enc = TimeEncode(time_dim)
        self.dropout = dropout
        self.n_smp_layers = n_smp_layers

        # TMP block
        self.tmp = TMPConv(in_channels=in_channels, time_dim=time_dim)

        # linear projection to hidden dimension for SMP block
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        # SMP blocks: L layers of static heterophily-aware message passing
        self.smp_convs = torch.nn.ModuleList(
            [SMPConv(hidden_channels=hidden_channels) for _ in range(n_smp_layers)]
        )

        self.smp_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_channels) for _ in range(n_smp_layers)]
        )

        # classifier MLP - same structure as static GNN baselines
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

    def forward(
            self,
            x:          torch.Tensor,   # [N_sub, in_channels]
            edge_index: torch.Tensor,   # [2, E_sub]
            time:       torch.Tensor,   # [E_sub, 1]   raw edge timestamps  (graph.time)
            node_time:  torch.Tensor,   # [N_sub]      node timestamps       (
            batch_size: int,
    ) -> torch.Tensor:              # [batch_size]  logits

        # time encoding 
        # node_time[dst] is the query time for each destination node
        # rel_t = query_time[dst] - edge_time (= t\'_dst - t_edge)
        dst = edge_index[1]
        rel_t = node_time[dst] - time
        rel_t_enc = self.time_enc(rel_t.float().unsqueeze(1)).squeeze(1)  # [E, time_dim]

        # TMP block
        h = self.tmp(x, edge_index, rel_t_enc)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # project to hidden dim before SMP
        h = F.relu(self.input_proj(h))

        # SMP block
        for conv, norm in zip(self.smp_convs, self.smp_norms):
            h = norm(F.relu(conv(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

        # classify seed nodes only
        return self.clf(h[:batch_size]).squeeze(-1)               # [batch size]
    