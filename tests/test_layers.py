import pytest
import torch.nn.functional as F
import dgl
import torch
from graph_conn.layers.gcn_layer import GCNLayer
from graph_conn.layers.mlp_readout_layer import MLPReadout

def test_gcn_layer():
    g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
    g = dgl.add_self_loop(g)
    feat = torch.ones(6, 10)
    edge_weight = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 1, 1, 1, 1, 1, 1])
    g.edata['weight'] = edge_weight
    conv = GCNLayer(in_dim=10, out_dim=3, activation=F.relu, dropout=0, edge_norm=False, batch_norm=False)
    h = conv(g, feat, edge_weight)
    assert h.size()[0] == 6
    assert h.size()[1] == 3
    conv = GCNLayer(in_dim=10, out_dim=3, activation=F.relu, dropout=0, edge_norm=True, batch_norm=False)
    h = conv(g, feat, edge_weight)
    assert h.size()[0] == 6
    assert h.size()[1] == 3


def test_mlp_readout():
    input_dim = 10
    output_dim = 5
    input = torch.randn(10, input_dim)
    mlp = MLPReadout(input_dim, output_dim, L=0)
    out = mlp(input)
    assert True