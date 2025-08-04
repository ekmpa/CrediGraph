import os
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from tgrag.encoders.encoder import Encoder
from tgrag.utils.data_loading import load_edge_csv, load_node_csv


class TemporalDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        node_file: str = 'temporal_nodes.csv',
        edge_file: str = 'temporal_edges.csv',
        encoding: Optional[Dict[str, Encoder]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.node_file = node_file
        self.edge_file = edge_file
        self.idx_dict: Dict[str, List[int]]
        self.encoding = encoding
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.node_file, self.edge_file]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self) -> None:
        pass

    def process(self) -> None:
        node_path = os.path.join(self.raw_dir, self.node_file)
        edge_path = os.path.join(self.raw_dir, self.edge_file)
        x_full, mapping = load_node_csv(
            path=node_path,
            index_col=1,  # 'node_id'
            encoders=self.encoding,
        )

        if x_full is None:
            raise TypeError('X is None type. Please use an encoding.')

        df = pd.read_csv(node_path)
        df = df.set_index('node_id').loc[mapping.keys()]

        # Transductive nodes only:
        labeled_mask = (df['cr_score'] != -1.0).values
        cr_score = torch.tensor(df['cr_score'].values, dtype=torch.float).unsqueeze(
            1
        )  # CR_SCORE Shape? Does it include all the nodes
        edge_index, edge_attr = load_edge_csv(
            path=edge_path, src_index_col='src', dst_index_col='dst', encoders=None
        )

        # adj_t = to_torch_csr_tensor(edge_index, size=(x_full.size(0), x_full.size(0)))

        data = Data(x=x_full, y=cr_score, edge_index=edge_index, edge_attr=edge_attr)
        # data.adj_t = adj_t

        data.labeled_mask = torch.tensor(labeled_mask, dtype=torch.bool)

        labeled_idx = torch.nonzero(torch.tensor(labeled_mask), as_tuple=True)[0]
        labeled_scores = cr_score[labeled_idx].squeeze().numpy()

        thresholds = [0.0, 1 / 3, 2 / 3, 1.0]
        train_idx, valid_idx, test_idx = [], [], []

        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i], thresholds[i + 1]
            mask = (labeled_scores > lower) & (labeled_scores <= upper)
            bucket_indices = labeled_idx[mask]

            perm = torch.randperm(bucket_indices.size(0))
            bucket_indices = bucket_indices[perm]

            n = bucket_indices.size(0)
            train_end = int(0.6 * n)
            valid_end = int(0.8 * n)

            train_idx.append(bucket_indices[:train_end])
            valid_idx.append(bucket_indices[train_end:valid_end])
            test_idx.append(bucket_indices[valid_end:])

        train_idx = torch.cat(train_idx)
        valid_idx = torch.cat(valid_idx)
        test_idx = torch.cat(test_idx)

        # Set global indices for our transductive nodes:
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.valid_mask[valid_idx] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        self.idx_dict = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }

        assert data.edge_index.max() < data.x.size(0), 'edge_index out of bounds'

        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        if self.idx_dict is not None:
            return self.idx_dict
        raise TypeError('idx split is empty.')
