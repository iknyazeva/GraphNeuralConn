import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .layers.gcn_layer import GCNLayer
from graph_conn.layers.gin_layer import GINLayer
from .layers.mlp_readout_layer import MLPReadout
import attr
from pathlib import Path
import json


@attr.s(slots=True)
class NetParams:
    n_nodes = attr.ib(default=10)
    in_dim = attr.ib(default=2)
    hidden_dim = attr.ib(default=10)
    out_dim = attr.ib(default=10)
    n_classes = attr.ib(default=2)
    dropout = attr.ib(default=0)
    in_dropout = attr.ib(default=0)
    batch_norm = attr.ib(default=True)
    edge_norm = attr.ib(default=True)
    readout = attr.ib(default="mean")
    gin_neighbor_aggr_type = attr.ib(default="mean")
    gin_n_mlp_layers = attr.ib(default=1)
    L = attr.ib(default=2)
    n_layers = attr.ib(default=3)
    residual = attr.ib(default=True)
    n_epochs = attr.ib(default=5)
    init_lr = attr.ib(default=5e-5)
    lr_reduce_factor = attr.ib(default=0.5)
    lr_schedule_patience = attr.ib(default=25)
    min_lr = attr.ib(default=1e-6)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=20)
    test_size = attr.ib(default=0.1)
    val_size = attr.ib(default=0.1)
    model = attr.ib(default="GCNNet")

    @classmethod
    def from_file(cls, path_to_json: Path):
        """loads hyper params from json file"""
        with open(path_to_json) as json_file:
            params = json.loads(json_file.read())
            fields = {field.name for field in attr.fields(NetParams)}
            return cls(**{k: v for k, v in params.items() if k in fields})


class GNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        self.in_dim = net_params.in_dim
        self.n_nodes = net_params.n_nodes
        self.hidden_dim = net_params.hidden_dim
        self.out_dim = net_params.out_dim
        self.n_classes = net_params.n_classes
        self.in_dropout = net_params.in_dropout
        self.dropout = net_params.dropout
        self.L = net_params.L
        self.readout = net_params.readout
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual
        self.n_layers = net_params.n_layers

        self.in_feat_dropout = nn.Dropout(self.in_dropout)

        if self.readout == 'flatten':
            self.MLP_layer = MLPReadout(self.out_dim * self.n_nodes, self.n_classes, L=self.L)
        else:
            self.MLP_layer = MLPReadout(self.out_dim, self.n_classes, L=self.L)

    def forward(self, g, h, e):
        h = self.in_feat_dropout(h)

        for conv in self.layers:
            h = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "flatten":
            hg = g.ndata['h'].reshape(g.batch_size, -1)
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)


class GINNet(GNet):
    def __init__(self, net_params):
        super().__init__(net_params)

        n_mlp_layers = net_params.gin_n_mlp_layers
        neighbor_aggr_type = net_params.gin_neighbor_aggr_type

        self.layers = nn.ModuleList([GINLayer(self.in_dim, self.hidden_dim, self.hidden_dim, n_mlp_layers,
                                              neighbor_aggr_type, self.dropout, self.batch_norm,
                                              residual=self.residual)])
        self.layers.extend([GINLayer(self.hidden_dim, self.hidden_dim, self.hidden_dim, n_mlp_layers,
                                     neighbor_aggr_type, self.dropout, self.batch_norm, residual=self.residual) for _ in
                            range(self.n_layers - 1)])
        self.layers.append(GINLayer(self.hidden_dim, self.hidden_dim, self.out_dim, n_mlp_layers, neighbor_aggr_type,
                                    self.dropout, self.batch_norm, residual=self.residual))


class GCNNet(GNet):
    def __init__(self, net_params):
        super().__init__(net_params)

        self.edge_norm = net_params.edge_norm

        self.layers = nn.ModuleList([GCNLayer(self.in_dim, self.hidden_dim, F.relu,
                                              self.dropout, self.batch_norm, False)])
        self.layers.extend([GCNLayer(self.hidden_dim, self.hidden_dim, F.relu, self.dropout, self.batch_norm,
                                     edge_norm=self.edge_norm, residual=self.residual) for _ in
                            range(self.n_layers - 1)])
        self.layers.append(GCNLayer(self.hidden_dim, self.out_dim, F.relu, self.dropout, self.batch_norm,
                                    edge_norm=self.edge_norm, residual=self.residual))
