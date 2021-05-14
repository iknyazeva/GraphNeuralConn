import numpy as np


def compute_node_strength(sym_matrix):
    positive_edges = np.clip(sym_matrix, 0, sym_matrix.max())
    negative_edges = np.clip(sym_matrix, sym_matrix.min(), 0)
    node_strength_positive = np.sum(np.abs(positive_edges), axis=0)
    node_strength_positive /= np.max(node_strength_positive)
    node_strength_negative = np.sum(np.abs(negative_edges), axis=0)
    node_strength_negative /= np.max(node_strength_negative)
    return np.vstack([node_strength_positive, node_strength_negative]).T


def return_roi_conns(sym_matrix):
    return sym_matrix


def compute_roi_pos(roi_coord, ref_roi):
    pass
