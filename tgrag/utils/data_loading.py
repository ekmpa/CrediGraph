import gzip
from typing import IO, Callable, Dict, List, Set, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


def read_vertex_file(path: str) -> Set[str]:
    result: Set[str] = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            result.add(line.strip())
    return result


def read_edge_file(path: str, old_to_new: Dict[int, int]) -> Set[str]:
    result: Set[str] = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            src, dst = int(parts[0]), int(parts[1])
            src = old_to_new.get(src, src)
            dst = old_to_new.get(dst, dst)
            result.add(f'{src}\t{dst}')
    return result


def open_file(path: str) -> Callable[..., IO]:
    return gzip.open if path.endswith('.gz') else open


def load_node_csv(
    path: str, index_col: int, encoders: Dict | None = None
) -> Tuple[Tensor | None, Dict]:
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = []
        for key, encoder in encoders.items():
            if key in df.columns:
                xs.append(encoder(df[key].values))
            else:
                # Global encoder (In our case the RNIEncoder)
                xs.append(encoder(df.shape[0]))

        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    encoders: Dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    usecols = [src_index_col, dst_index_col]
    if encoders is not None:
        usecols += [col for col in encoders if col not in usecols]

    df = pd.read_csv(path, usecols=usecols)

    src = torch.tensor(df[src_index_col].to_numpy(), dtype=torch.long)
    dst = torch.tensor(df[dst_index_col].to_numpy(), dtype=torch.long)
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


def load_edges(edge_file: str) -> List[Tuple[str, str]]:
    edges = []
    with open_file(edge_file)(edge_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading edge file'):
            parts = line.strip().split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


def load_node_domain_map(node_file: str) -> Tuple[dict, dict]:
    id_to_domain = {}
    domain_to_id = {}
    with open_file(node_file)(node_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading node file'):
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, domain = parts
                id_to_domain[node_id] = domain
                domain_to_id[domain] = node_id
    return id_to_domain, domain_to_id
