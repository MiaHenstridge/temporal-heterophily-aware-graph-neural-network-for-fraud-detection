import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, TransformerConv
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
        # rel_t = node_time[src] - edge_time  →  [E, 1]
        rel_t     = node_time[edge_index[0]].view(-1, 1) - time   # [E, 1]
        rel_t_enc = self.time_enc(rel_t).squeeze(1)               # [E, time_dim]

        # ── input projection ──────────────────────────────────────────────────
        h = F.relu(self.input_proj(x))                            # [N, hidden]

        # ── conv layers ───────────────────────────────────────────────────────
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index, edge_attr=rel_t_enc).relu())
            h = F.dropout(h, p=self.dropout, training=self.training)

        # ── classify seed nodes only ──────────────────────────────────────────
        return self.clf(h[:batch_size]).squeeze(-1)               # [batch_size]