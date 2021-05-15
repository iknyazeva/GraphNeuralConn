from graph_conn.models import GCNNet, NetParams, GINNet
from dgl.dataloading import GraphDataLoader
import torch
from graph_conn.conn_dataset import GraphParams, CorrToDGLDataset
from graph_conn.conn_data_utils import create_test_corr
import pytest


@pytest.fixture
def small_dataset():
    graph_params = GraphParams.from_file('../configs/test_graph_conn.json')
    N = 60
    create_test_corr(N, graph_params)
    small_dataset = CorrToDGLDataset(graph_params)
    return small_dataset

def test_net_params():
    net_params = NetParams.from_file("../configs/test_gcn.json")
    assert isinstance(net_params.in_dim, int)
    assert isinstance(net_params.residual, bool)
    net_params_gin = NetParams.from_file("../configs/test_gin.json")
    assert isinstance(net_params_gin.hidden_dim_dim, int)


def test_ginnet_one_graph(small_dataset):
    net_params = NetParams.from_file("../configs/test_gin.json")
    model = GINNet(net_params=net_params)
    graph, label = small_dataset[0]
    h = graph.ndata['feat']
    e = graph.edata['weight']
    out = model(graph, h, e)
    assert isinstance(h, torch.Tensor)
    assert isinstance(e, torch.Tensor)
    assert 1 == out.shape[0]


def test_gcnnet_one_graph(small_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    model = GCNNet(net_params=net_params)
    graph, label = small_dataset[0]
    h = graph.ndata['feat']
    e = graph.edata['weight']
    out = model(graph, h, e)
    assert isinstance(h, torch.Tensor)
    assert isinstance(e, torch.Tensor)
    assert 1 == out.shape[0]


def test_gcnnet_batched_graph(small_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    #net_params.n_nodes = small_dataset[0][0].number_of_nodes()
    model = GCNNet(net_params=net_params)
    dataloader = GraphDataLoader(small_dataset, batch_size=3)
    batched_graph, labels = next(iter(dataloader))
    h = batched_graph.ndata['feat']
    e = batched_graph.edata['weight']
    out = model(batched_graph, h, e)
    assert True

def test_gcnnet_batched_graph_flatten(small_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    net_params.n_nodes = small_dataset[0][0].number_of_nodes()
    net_params.readout = 'flatten'
    model = GCNNet(net_params=net_params)
    dataloader = GraphDataLoader(small_dataset, batch_size=3)
    batched_graph, labels = next(iter(dataloader))
    h = batched_graph.ndata['feat']
    e = batched_graph.edata['weight']
    out = model(batched_graph, h, e)
    assert True