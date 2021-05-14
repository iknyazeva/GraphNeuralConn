import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""


class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, edge_norm=False, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edge_norm = edge_norm
        self.batch_norm = batch_norm
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if edge_norm:
            self.conv = GraphConv(in_dim, out_dim, norm='none', weight=True, bias=True, allow_zero_in_degree=True)
        else:
            self.conv = GraphConv(in_dim, out_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True)

    def forward(self, g, h, e):
        # e = g.edata['weight']
        h_in = h  # to be used for residual connection

        if self.edge_norm:
            norm = EdgeWeightNorm(norm='both')
            e = norm(g, e)

        h = self.conv(g, h, edge_weight=e)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)
