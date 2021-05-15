from graph_conn.conn_dataset import FullAbideDataset, CorrToDGLDataset
from graph_conn.conn_dataset import GraphParams
from graph_conn.conn_data_utils import dgl_graph_from_vec, create_test_corr
import numpy as np
import pickle
import dgl
import torch
import pytest


@pytest.fixture
def graph_params():
    graph_params = GraphParams()
    return graph_params

def test_graph_params():
    graph_params = GraphParams.from_file('../configs/test_graph_conn.json')
    assert isinstance(graph_params.cut_type, bool)
    assert isinstance(graph_params.n_nodes, int)

def test_corr_to_dgl_dataset():
    graph_params = graph_params = GraphParams.from_file('../configs/test_graph_conn.json')
    N = 60
    create_test_corr(N, graph_params)
    dataset = CorrToDGLDataset(graph_params)
    split = dataset.get_split_idx()
    assert isinstance(split['train'], torch.LongTensor)
    assert isinstance(dataset[0][0], dgl.DGLGraph)


def test_full_abide_dataset():
    graph_params = GraphParams.from_file('../configs/abide_graph_conn.json')
    dataset = CorrToDGLDataset(graph_params)
    split = dataset.get_split_idx()
    test_dataset = dataset[split['test']]
    assert isinstance(split['train'], torch.LongTensor)
    assert isinstance(dataset[0][0], dgl.DGLGraph)


def test_list_to_dgl_dataset():
    graph_params = GraphParams()
    graph_params.n_nodes = 10
    N = 5
    graphs = []
    for i in range(N):
        sym = 0.3 * np.random.randn(graph_params.n_nodes, graph_params.n_nodes)
        graphs.append(dgl_graph_from_vec(sym, graph_params, flatten=False))
    labels = torch.LongTensor(np.random.choice([0, 1], size=N, p=[0.3, 0.7]))
    test_dataset = ListToDGLDataset(graphs, labels)
    graph, label = test_dataset[0]
    assert isinstance(graph, dgl.DGLGraph)
    assert isinstance(label, torch.LongTensor)