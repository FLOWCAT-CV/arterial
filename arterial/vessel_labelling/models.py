#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ResGAT(nn.Module):
    """
    GATv2 with a residual (root) connection in every layer: h = GATv2Conv(x) + W x.

    In the node-form (line) graph, all segments meeting at a bifurcation form a clique, so
    sibling segments (e.g. ICA and ECA) share the same neighbourhood. Plain GAT has no separate
    path for a node's own features, so siblings get almost the same representation; the residual
    term keeps them apart. Equivalent to GATv2Conv(residual=True) in PyG >= 2.6. Parameter names
    match the training implementation (EVC repository), so its state dicts load strictly.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output classes.
    num_layers : int
        Number of message-passing layers.
    dropout : float, optional
        Dropout applied after each hidden layer. The default is 0.0.
    act : callable, optional
        Activation function. The default is F.leaky_relu.
    norm : string, optional
        None or "batch_norm", applied after each hidden layer (before the activation). The
        default is None.

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout = 0.0, act = F.leaky_relu, norm = None):
        """
        Initializes the GATv2 and residual layers.

        """
        super().__init__()
        self.in_channels, self.hidden_channels, self.out_channels, self.num_layers = in_channels, hidden_channels, out_channels, num_layers
        self.dropout, self.act, self.norm = dropout, act, norm
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList([GATv2Conv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.res = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias = False) for i in range(num_layers)])
        if norm not in (None, "batch_norm"):
            raise ValueError(f"Unknown norm {norm}.")
        self.norms = None if norm is None else nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index, *args, **kwargs):
        """
        Computes node logits.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge index, shape (2, num_edges).

        Returns
        -------
        x : torch.Tensor
            Node logits, shape (num_nodes, out_channels).

        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) + self.res[i](x)
            if i < self.num_layers - 1:
                if self.norms is not None:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p = self.dropout, training = self.training)
        return x
