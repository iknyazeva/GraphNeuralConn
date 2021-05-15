import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GINConv

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""


class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, aggr_type, dropout, batch_norm, residual=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        if in_dim != out_dim:
            self.residual = False

        self.bn_node_h = nn.BatchNorm1d(out_dim)
        mlp = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.conv = GINConv(mlp, aggr_type, init_eps=init_eps, learn_eps=learn_eps)

    def forward(self, g, h, e):
        h_in = h  # for residual connection

        h = self.conv(g, h, edge_weight=e)

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        h = F.relu(h)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection

        h = F.dropout(h, self.dropout)

        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
