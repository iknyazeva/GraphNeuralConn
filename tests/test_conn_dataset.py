from graph_conn.conn_dataset import FullAbideDataset, ListToDGLDataset
from graph_conn.conn_dataset import GraphParams
from graph_conn.conn_data_utils import dgl_graph_from_vec
import numpy as np
import dgl
import torch
import pytest


@pytest.fixture
def graph_params():
    graph_params = GraphParams()
    return graph_params

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

def test_full_abide_dataset():
    graph_params = GraphParams()
    dataset = FullAbideDataset(graph_params)
    split = dataset.get_split_idx()
    test_dataset = dataset[split['test']]
    assert True
