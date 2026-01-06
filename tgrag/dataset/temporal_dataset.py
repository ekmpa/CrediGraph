import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset

from tgrag.encoders.encoder import Encoder
from tgrag.utils.readers import (
    get_full_dict,
    load_large_edge_csv,
    load_node_csv,
)
from tgrag.utils.target_generation import generate_exact_targets_csv


class TemporalDataset(InMemoryDataset):
    """Graph dataset with temporal / versioned preprocessing and target generation.

    Loads raw node, edge, and target files, generates targets if missing,
    constructs PyTorch Geometric Data objects, and creates stratified
    train/validation/test splits over labeled nodes.
    """

    def __init__(
        self,
        root: str,
        node_file: str = 'features.csv',
        edge_file: str = 'edges.csv',
        target_file: str = 'target.csv',
        target_col: str = 'score',
        target_index_name: str = 'nid',
        target_index_col: int = 0,
        edge_src_col: str = 'src',
        edge_dst_col: str = 'dst',
        index_col: int = 1,
        index_name: str = 'node_id',
        encoding: Optional[Dict[str, Encoder]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 42,
        processed_dir: Optional[str] = None,
    ):
        """Initialize the dataset configuration and load processed data if present.

        Parameters:
            root : str
                Root directory containing raw and processed dataset files.
            node_file : str
                Node feature CSV file name.
            edge_file : str
                Edge CSV file name.
            target_file : str
                Target label CSV file name.
            target_col : str
                Column name containing target values.
            target_index_name : str
                Name of the index column in the target file.
            target_index_col : int
                Column index of the target index if no header is present.
            edge_src_col : str
                Column name for source node IDs in the edge file.
            edge_dst_col : str
                Column name for destination node IDs in the edge file.
            index_col : int
                Column index of node IDs in the node file.
            index_name : str
                Name of the node ID column.
            encoding : Optional[Dict[str, Encoder]]
                Optional encoders applied to node feature columns.
            transform : Optional[Callable]
                Optional transform applied on each access.
            pre_transform : Optional[Callable]
                Optional transform applied before saving processed data.
            seed : int
                Random seed for dataset splitting.
            processed_dir : Optional[str]
                Optional override for processed data directory.
        """
        self.node_file = node_file
        self.edge_file = edge_file
        self.target_file = target_file
        self.target_col = target_col
        self.edge_src_col = edge_src_col
        self.edge_dst_col = edge_dst_col
        self.index_col = index_col
        self.index_name = index_name
        self.target_index_name = target_index_name
        self.target_index_col = target_index_col
        self.encoding = encoding
        self.seed = seed
        self._custome_processed_dir = processed_dir
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        """Return the directory containing raw dataset files."""
        return os.path.join(self.root)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of expected raw file names."""
        return [self.node_file, self.edge_file]

    @property
    def processed_dir(self) -> str:
        """Return the directory used to store processed dataset files."""
        if self._custome_processed_dir is not None:
            return self._custome_processed_dir
        return super().processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return ['data.pt']

    def download(self) -> None:
        """No-op download hook (raw data must already exist locally)."""

    def process(self) -> None:
        """Generate targets, construct graph tensors, and create train/valid/test splits."""
        node_path = os.path.join(self.raw_dir, self.node_file)
        edge_path = os.path.join(self.raw_dir, self.edge_file)
        target_path = os.path.join(self.raw_dir, self.target_file)
        if os.path.exists(target_path):
            logging.info('Target file already exists.')
        else:
            logging.info('Generating target file.')
            dqr = get_full_dict()
            generate_exact_targets_csv(node_path, target_path, dqr)

        logging.info('***Constructing Feature Matrix***')
        x_full, mapping, full_index = load_node_csv(
            path=node_path,
            index_col=0,
            encoders=self.encoding,
        )
        logging.info('***Feature Matrix Done***')

        if x_full is None:
            raise TypeError('X is None type. Please use an encoding.')

        df_target = pd.read_csv(target_path)
        logging.info(f'Size of target dataframe: {df_target.shape}')

        mapping_index = [mapping[domain.strip()] for domain in df_target['domain']]
        df_target.index = mapping_index
        logging.info(f'Size of mapped target dataframe: {df_target.shape}')

        missing_idx = full_index.difference(mapping_index)
        filler = pd.DataFrame(
            {col: np.nan for col in df_target.columns}, index=missing_idx
        )
        df_target = pd.concat([df_target, filler])
        df_target.sort_index(inplace=True)
        logging.info(f'Size of filled target dataframe: {df_target.shape}')
        score = torch.tensor(
            df_target[self.target_col].astype('float32').fillna(-1).values,
            dtype=torch.float,
        )
        logging.info(f'Size of score vector: {score.size()}')

        labeled_mask = score != -1.0

        labeled_idx = torch.nonzero(torch.tensor(labeled_mask), as_tuple=True)[0]
        labeled_scores = score[labeled_idx].squeeze().numpy()

        if labeled_scores.size == 0:
            raise ValueError(
                f"No labeled nodes found in target column '{self.target_col}'"
            )

        logging.info('***Constructing Edge Matrix***')
        edge_index, edge_attr = load_large_edge_csv(
            path=edge_path,
            src_index_col=self.edge_src_col,
            dst_index_col=self.edge_dst_col,
            mapping=mapping,
            encoders=None,
        )
        logging.info('***Edge Matrix Constructed***')

        data = Data(x=x_full, y=score, edge_index=edge_index, edge_attr=edge_attr)

        data.labeled_mask = labeled_mask.detach().clone().bool()

        quantiles = np.quantile(labeled_scores, [1 / 3, 2 / 3])
        quartile_labels = np.digitize(labeled_scores, bins=quantiles)

        train_idx, temp_idx, _, quartile_labels_temp = train_test_split(
            labeled_idx,
            quartile_labels,
            train_size=0.6,
            stratify=quartile_labels,
            random_state=self.seed,
        )

        valid_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            stratify=quartile_labels_temp,
            random_state=self.seed,
        )

        train_idx = torch.as_tensor(train_idx)
        logging.info(f'Train size: {train_idx.size()}')
        valid_idx = torch.as_tensor(valid_idx)
        logging.info(f'Valid size: {valid_idx.size()}')
        test_idx = torch.as_tensor(test_idx)
        logging.info(f'Test size: {test_idx.size()}')

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

        torch.save(mapping, self.processed_dir + '/mapping.pt')
        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        """Return the stored train/valid/test index split.

        Returns:
            Dict
                Mapping with keys {'train', 'valid', 'test'} and index tensors.

        Raises:
            TypeError
                If the split is not available.
        """
        data = self[0]
        if hasattr(data, 'idx_dict') and data.idx_dict is not None:
            return data.idx_dict
        raise TypeError('idx split is empty.')

    def get_mapping(self) -> Dict:
        """Return the node ID mapping (lazy) from raw identifiers to internal indices."""
        if not hasattr(self, '_mapping'):
            self._mapping = torch.load(self.processed_dir + '/mapping.pt')
        return self._mapping
