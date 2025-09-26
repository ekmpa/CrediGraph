import logging
import pickle
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from tgrag.utils.path import get_root_dir


def load_node_csv(
    path: str, index_col: int, encoders: Dict | None = None, chunk_size: int = 500_000
) -> Tuple[Tensor | None, Dict]:
    dfs = []
    total_rows = sum(1 for _ in open(path)) - 1
    with pd.read_csv(path, index_col=index_col, chunksize=chunk_size) as reader:
        for chunk in tqdm(
            reader, total=total_rows // chunk_size + 1, desc='Reading node CSV'
        ):
            dfs.append(chunk)

    df = pd.concat(dfs, axis=0)
    mapping = {
        index: i for i, index in tqdm(enumerate(df.index.unique()), desc='Indexing')
    }

    x = None
    if encoders is not None:
        xs = []
        for key, encoder in encoders.items():
            if key in df.columns:
                xs.append(encoder(df[key].values))
            elif key == 'pre':
                logging.info('Pre-constructed text embeddings used.')
                xs.append(encoder(df['domain'], get_seed_embeddings()))
            else:
                xs.append(encoder(df.shape[0]))

        if len(xs) == 1:
            x = xs[0]
        else:
            x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    mapping: Dict,
    encoders: Dict | None = None,
    chunk_size: int = 500_000,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    usecols = [src_index_col, dst_index_col]
    if encoders is not None:
        usecols += [col for col in encoders if col not in usecols]

    dfs = []
    total_rows = sum(1 for _ in open(path)) - 1
    with pd.read_csv(path, usecols=usecols, chunksize=chunk_size) as reader:
        for chunk in tqdm(
            reader, total=total_rows // chunk_size + 1, desc='Reading edge CSV'
        ):
            dfs.append(chunk)

    df = pd.concat(dfs, axis=0)

    src = torch.tensor([mapping[s] for s in df[src_index_col]], dtype=torch.long)
    dst = torch.tensor([mapping[d] for d in df[dst_index_col]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)  # Shape: [2, num_edges]

    edge_attr = None
    if encoders is not None:
        edge_attrs = []
        for key, encoder in encoders.items():
            if key in df.columns:
                edge_attrs.append(encoder(df[key]))
            else:
                edge_attrs.append(encoder(len(df)))

        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def load_large_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    mapping: Dict,
    encoders: Dict | None = None,
    chunk_size: int = 500_000,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    usecols = [src_index_col, dst_index_col]
    if encoders is not None:
        usecols += [col for col in encoders if col not in usecols]

    src_list, dst_list = [], []
    edge_attrs_list = []

    total_rows = sum(1 for _ in open(path)) - 1
    with pd.read_csv(path, usecols=usecols, chunksize=chunk_size) as reader:
        for chunk in tqdm(
            reader, total=total_rows // chunk_size + 1, desc='Reading edge CSV'
        ):
            # Map src/dst indices to integer ids
            src_list.append(
                torch.tensor(
                    [mapping[s] for s in chunk[src_index_col].values], dtype=torch.long
                )
            )
            dst_list.append(
                torch.tensor(
                    [mapping[d] for d in chunk[dst_index_col].values], dtype=torch.long
                )
            )

            # Encode edge attributes (if any)
            if encoders is not None:
                edge_attrs = []
                for key, encoder in encoders.items():
                    if key in chunk.columns:
                        edge_attrs.append(encoder(chunk[key].values))
                    else:
                        edge_attrs.append(encoder(len(chunk)))
                edge_attrs_list.append(torch.cat(edge_attrs, dim=-1))

    # Concatenate results from all chunks
    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr = None
    if edge_attrs_list:
        edge_attr = torch.cat(edge_attrs_list, dim=0)

    return edge_index, edge_attr


def get_seed_embeddings() -> Dict[str, torch.Tensor]:
    root = get_root_dir()
    path = f'{root}/data/dqr/labeled_11k_domainname_emb/labeled_11k_domainName_emb.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)

    embeddings_lookup = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()
    }

    return embeddings_lookup
