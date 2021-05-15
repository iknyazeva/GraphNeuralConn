import numpy as np
import dgl
import torch
import os
from scipy import sparse
from tqdm import tqdm
import pickle
from graph_conn import feature_generation



def create_test_corr(N, graph_params):
    """

    Args:
        N (int): number of matrix
        graph_params (GraphParams): parameters of processing

    Returns:

    """
    matrix = 0.3 * np.random.randn(N, graph_params.n_nodes, graph_params.n_nodes)
    labels = np.random.choice([0, 1], size=N, p=[0.3, 0.7])
    conn = {'correlation': matrix, 'labels': labels}
    with open(os.path.join(graph_params.raw_dir, graph_params.filename), 'wb') as handle:
        pickle.dump(conn, handle, protocol=pickle.DEFAULT_PROTOCOL)



def sym_to_vec(symmetric):
    """

    Args:
        symmetric (numpy.ndarray): input array of  symmetric matrices with shape (..., n_features, n_features)

    Returns:
        numpy.ndarray: The output flattened lower triangular part of symmetric. Shape is
        (..., n_features-1 * n_features / 2)
    """
    tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(np.bool)
    return symmetric[..., tril_mask]


def vec_to_sym(vec, Nnodes=264):
    """

    Args:
        vec (numpy.ndarray): The output flattened lower triangular part of symmetric. Shape is
        (..., n_features-1 * n_features / 2)
        Nnodes (int): number of nodes in full matrix

    Returns:
        (numpy.ndarray): input array of  symmetric matrices with shape (..., n_features, n_features)
    """
    tril_mask = np.tril(np.ones((Nnodes, Nnodes)), k=-1).astype(bool)
    if len(vec.shape) == 1:
        vec = vec.reshape(1, -1)
    sym = np.zeros((vec.shape[0], Nnodes, Nnodes))
    sym[:, tril_mask] = vec
    return np.squeeze(sym + np.transpose(sym, axes=(0, 2, 1)))


def compute_node_strength(sym_matrix):
    positive_edges = np.clip(sym_matrix, 0, sym_matrix.max())
    negative_edges = np.clip(sym_matrix, sym_matrix.min(), 0)
    node_strength_positive = np.sum(np.abs(positive_edges), axis=0)
    node_strength_positive /= np.max(node_strength_positive)
    node_strength_negative = np.sum(np.abs(negative_edges), axis=0)
    node_strength_negative /= np.max(node_strength_negative)
    return np.vstack([node_strength_positive, node_strength_negative]).T


def dgl_graph_from_vec(vec, graph_params):
    """
    Create graph from flatten vector as a thresholed weighted matrix with properties
    as type torch 
    """

    if graph_params.flatten:
        W = vec_to_sym(vec)
    else:
        W = vec
    # create graph
    # add signal on nodes
    u = getattr(feature_generation, graph_params.node_feat)(W)
    if graph_params.thr_type == 'pos':
        W[W < graph_params.threshold] = 0
    elif graph_params.thr_type == 'both':
        W[np.abs(W) < graph_params.threshold] = 0
    elif graph_params.thr_type == 'neg':
        W[-W < graph_params.threshold] = 0
    else:
        W[W < graph_params.threshold] = 0



    # convert to pytorch?
    W = sparse.csr_matrix(W).tocoo()
    edge_weight = torch.tensor(W.data).float()
    u = torch.from_numpy(u.astype(np.float32))

    g = dgl.from_scipy(W)
    g.ndata['feat'] = u
    g.edata['weight'] = edge_weight

    if graph_params.add_self_loop:
        g = dgl.add_self_loop(g)
        g.edata['weight'][-graph_params.n_nodes:] = 1
    return g


def load_data_to_graphs(graph_params):
    graphs = []
    labels = []
    with open(os.path.join(graph_params.raw_dir, graph_params.filename), 'rb') as handle:
        conn = pickle.load(handle)
    for row in tqdm(conn[graph_params.corr_name]):
        graphs.append(dgl_graph_from_vec(row, graph_params))
        if graph_params.name == "power_abide":
            labels = [item // 2 for item in conn[graph_params.target_name]]
        else:
            labels = conn[graph_params.target_name]
        # Convert the label list to tensor for saving.
        labels = torch.LongTensor(labels)
    return graphs, labels


