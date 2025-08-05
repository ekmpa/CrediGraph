import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
        seed: int = 42,
    ):
        self.node_file = node_file
        self.edge_file = edge_file
        self.encoding = encoding
        self.seed = seed
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

        quantiles = np.quantile(labeled_scores, [1 / 3, 2 / 3])
        quartile_labels = np.digitize(labeled_scores, bins=quantiles)

        train_idx, temp_idx, _, _ = train_test_split(
            labeled_idx,
            quartile_labels,
            train_size=0.6,
            stratify=quartile_labels,
            random_state=self.seed,
        )

        temp_idx_np = (
            temp_idx.numpy() if isinstance(temp_idx, torch.Tensor) else temp_idx
        )

        quartile_labels_temp = quartile_labels[
            np.isin(labeled_idx.numpy(), temp_idx_np)
        ]

        valid_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            stratify=quartile_labels_temp,
            random_state=self.seed,
        )

        train_idx = torch.as_tensor(train_idx)
        valid_idx = torch.as_tensor(valid_idx)
        test_idx = torch.as_tensor(test_idx)

        # Set global indices for our transductive nodes:
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.valid_mask[valid_idx] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        data.idx_dict = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }

        assert data.edge_index.max() < data.x.size(0), 'edge_index out of bounds'

        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        data = self[0]
        if hasattr(data, 'idx_dict') and data.idx_dict is not None:
            return data.idx_dict
        raise TypeError('idx split is empty.')
