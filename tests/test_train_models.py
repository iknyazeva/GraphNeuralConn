from graph_conn.train_models import ConnGCM
from graph_conn.models import NetParams
from graph_conn.conn_dataset import GraphParams, CorrToDGLDataset
from graph_conn.conn_data_utils import create_test_corr
import pytest


@pytest.fixture
def medium_dataset():
    graph_params = GraphParams.from_file('../configs/test_graph_conn.json')
    N = 100
    create_test_corr(N, graph_params)
    medium_dataset = CorrToDGLDataset(graph_params)
    return medium_dataset
@pytest.fixture
def medium_roi_dataset():
    graph_params = GraphParams.from_file('../configs/test_graph_conn.json')
    graph_params.n_nodes = 20
    graph_params.thr_type = 'both'
    graph_params.node_feat = 'return_roi_conns'
    N = 50
    create_test_corr(N, graph_params)
    medium_roi_dataset = CorrToDGLDataset(graph_params)
    return medium_roi_dataset

def test_gcn_net(medium_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    net_params.n_nodes = medium_dataset[0][0].number_of_nodes()
    net_params.readout = 'flatten'
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_dataset)
    assert True

def test_roi_gcn_net(medium_roi_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    net_params.n_nodes = medium_roi_dataset[0][0].number_of_nodes()
    net_params.in_dim = net_params.n_nodes
    net_params.readout = 'flatten'
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_roi_dataset)
    assert True

def test_gcn_net_flatten(medium_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    net_params.n_nodes = medium_dataset[0][0].number_of_nodes()
    net_params.readout = 'flatten'
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_dataset)
    assert True

def test_gin_net(medium_dataset):
    net_params = NetParams.from_file("../configs/test_gin.json")
    net_params.n_nodes = medium_dataset[0][0].number_of_nodes()
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_dataset,  scheduler=True)
    assert True

def test_gcn_abide():
    graph_params = GraphParams.from_file('../configs/abide_graph_conn.json')
    dataset = CorrToDGLDataset(graph_params)
    net_params = NetParams.from_file("../configs/abide_gcn.json")
    net_params.n_epochs = 2
    model = ConnGCM(net_params)
    history = model.train(dataset=dataset, scheduler=False)
    assert True


def test_gin_abide():
    graph_params = GraphParams.from_file('../configs/abide_graph_conn.json')
    dataset = CorrToDGLDataset(graph_params)
    net_params = NetParams.from_file("../configs/abide_gin.json")
    net_params.n_epochs = 2
    model = ConnGCM(net_params)
    history = model.train(dataset=dataset, scheduler=False)
    assert True





