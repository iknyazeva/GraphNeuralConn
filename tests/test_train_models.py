from graph_conn.models import GCNNet, NetParams
from graph_conn.train_models import ConnGCM
from graph_conn.conn_dataset import FullAbideDataset, ListToDGLDataset
from graph_conn.conn_data_utils import dgl_graph_from_vec
from graph_conn.conn_dataset import GraphParams
import pytest
import numpy as np
from graph_conn.models import GCNNet, NetParams
from dgl.dataloading import GraphDataLoader
import torch
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

def test_gcn_net(medium_dataset):
    net_params = NetParams.from_file("../configs/test_gcn.json")
    #net_params.n_nodes = medium_dataset[0][0].number_of_nodes()
    #net_params.readout = 'flatten'
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_dataset)
    assert True


