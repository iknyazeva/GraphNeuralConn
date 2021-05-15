import pytest
import torch.nn.functional as F
import dgl
import torch
from graph_conn.layers.gcn_layer import GCNLayer
from graph_conn.layers.gin_layer import GINLayer, MLP
from graph_conn.layers.mlp_readout_layer import MLPReadout


@pytest.fixture
def g():
    g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
    g = dgl.add_self_loop(g)
    feat = torch.ones(6, 10)
    edge_weight = torch.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 1, 1, 1, 1, 1, 1])
    g.ndata['feat'] = feat
    g.edata['weight'] = edge_weight
    return g

def test_gin_layer(g):
    g = dgl.transform.remove_self_loop(g)
    feat = g.ndata['feat']
    edge_weight = g.edata['weight']
    in_dim = feat.shape[1]
    hidden_dim = 10
    num_layers = 1
    out_dim = 7
    neighbor_aggr_type = "mean"
    dropout = 0.0
    batch_norm = True
    conv = GINLayer(in_dim, hidden_dim, out_dim, num_layers, neighbor_aggr_type, dropout, batch_norm)
    h = conv(g, feat, edge_weight)
    assert h.size()[0] == 6
    assert h.size()[1] == 7
    num_layers = 2
    conv = GINLayer(in_dim, hidden_dim, out_dim, num_layers, neighbor_aggr_type, dropout, batch_norm)

    h = conv(g, feat, edge_weight)
    assert h.size()[0] == 6
    assert h.size()[1] == 7


def test_gcn_layer(g):
    feat = g.ndata['feat']
    edge_weight = g.edata['weight']
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