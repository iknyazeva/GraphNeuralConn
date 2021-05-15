import torch
import dgl
from graph_conn.conn_data_utils import load_data_to_graphs, dgl_graph_from_vec
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
from pathlib import Path
from dgl.data.utils import Subset

import attr
import json


# graph_params: attributed class with all the parameters
@attr.s(slots=True)
class GraphParams:
    n_nodes = attr.ib(default=264)
    flatten = attr.ib(default=False)
    threshold = attr.ib(default=0.10)
    thr_type = attr.ib(default='pos')
    cut_type = attr.ib(default=True)
    node_feat = attr.ib(default='compute_node_strength')
    raw_dir = attr.ib(default='../dataset/power_abide/')
    name = attr.ib(default='power_abide')
    add_self_loop = attr.ib(default=True)
    filename = attr.ib(default='power_conn_abide.pickle')
    corr_name = attr.ib(default='correlation')
    target_name = attr.ib(default='abide_diagnosis')

    @classmethod
    def from_file(cls, path_to_json: Path):
        """loads hyper params from json file"""
        with open(path_to_json) as json_file:
            params = json.loads(json_file.read())
            fields = {field.name for field in attr.fields(GraphParams)}
            return cls(**{k: v for k, v in params.items() if k in fields})


class CorrToDGLDataset(DGLDataset):

    def __init__(self, graph_params):
        self.graphs = []
        self.labels = []
        self.graph_params = graph_params
        super().__init__(name=graph_params.name, raw_dir=graph_params.raw_dir)

    def process(self):

        graphs, labels = load_data_to_graphs(self.graph_params)
        self.graphs = graphs
        self.labels = labels

    def get_split_idx(self, test_size=0.1, val_size=0.1):

        X = range(len(self.labels))
        y = self.labels.numpy()
        train_idx, test_idx, y_train, _ = train_test_split(X, y, stratify=y, test_size=test_size)
        train_idx, valid_idx, _, _ = train_test_split(train_idx, y_train, stratify=y_train, test_size=val_size)
        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        """
        Get datapoint with index
        Args:
            idx: int or array of ints

        Returns:
        """

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        return len(self.graphs)


class FullAbideDataset(CorrToDGLDataset):
    """
    download dataset from flatten structure saved as a pickle dictionary with keys corr_name and target_name
    """

    def __init__(self, graph_params):
        # self.graphs = []
        # self.labels = []
        # self.graph_params = graph_params
        super().__init__(name=graph_params.name, raw_dir=graph_params.raw_dir)

    def process(self):
        graphs, labels = load_data_to_graphs(self.graph_params)
        self.graphs = graphs
        self.labels = labels

    def get_split_idx(self, test_size=0.1, val_size=0.1):

        X = range(len(self.labels))
        y = self.labels.numpy()
        train_idx, test_idx, y_train, _ = train_test_split(X, y, stratify=y, test_size=test_size)
        train_idx, valid_idx, _, _ = train_test_split(train_idx, y_train, stratify=y_train, test_size=val_size)
        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        """ Get datapoint with index
        Args:
          idx: int or array of ints

        Returns:
        """

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        return len(self.graphs)


class ListToDGLDataset(DGLDataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        super().__init__(name='lists')
        # assert all(len(lists[0]) == len(li) for li in lists)

    def process(self):
        pass

    def get_split_idx(self, test_size=0.1, val_size=0.1):

        X = range(len(self.labels))
        y = self.labels.numpy()
        train_idx, test_idx, y_train, _ = train_test_split(X, y, stratify=y, test_size=test_size)
        train_idx, valid_idx, _, _ = train_test_split(train_idx, y_train, stratify=y_train, test_size=val_size)
        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        """
        Get datapoint with index
        Args:
            idx: int or array of ints

        Returns:
        """

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        return len(self.graphs)

