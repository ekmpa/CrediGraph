import os
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_torch_csr_tensor

from tgrag.utils.data_loading import load_edge_csv, load_node_csv


class TemporalDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        encoding: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.encoding = encoding
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root)

    @property
    def raw_file_names(self) -> List[str]:
        return ['temporal_nodes.csv', 'temporal_edges.csv']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self) -> None:
        pass

    def process(self) -> None:
        node_path = os.path.join(self.raw_dir, 'temporal_nodes.csv')
        edge_path = os.path.join(self.raw_dir, 'temporal_edges.csv')
        x_full, mapping = load_node_csv(
            path=node_path,
            index_col=1,  # 'node_id'
            encoders=self.encoding,
        )

        if x_full is None:
            raise TypeError('X is None type. Please use an encoding.')

        df = pd.read_csv(node_path)
        df = df.set_index('node_id').loc[mapping.keys()]
        labeled_mask = df['cr_score'].notna().values
        cr_score = torch.tensor(
            df['cr_score'].fillna(-1).values, dtype=torch.float
        ).unsqueeze(1)

        edge_index, edge_attr = load_edge_csv(
            path=edge_path, src_index_col='src', dst_index_col='dst', encoders=None
        )

        adj_t = to_torch_csr_tensor(edge_index, size=(x_full.size(0), x_full.size(0)))

        data = Data(x=x_full, y=cr_score, edge_index=edge_index, edge_attr=edge_attr)
        data.adj_t = adj_t

        data.labeled_mask = torch.tensor(labeled_mask, dtype=torch.bool)

        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        data = self[0]
        labeled_idx = data.labeled_mask.nonzero(as_tuple=True)[0]
        n = labeled_idx.size(0)
        train_end = int(0.6 * n)
        valid_end = int(0.8 * n)
        return {
            'train': labeled_idx[:train_end],
            'valid': labeled_idx[train_end:valid_end],
            'test': labeled_idx[valid_end:],
        }
