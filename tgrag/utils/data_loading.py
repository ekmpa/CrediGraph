import gzip
from datetime import date
from typing import IO, Callable, Dict, List, Set, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


def iso_week_to_timestamp(iso_week_str: str) -> str:
    """Convert CC-MAIN-YYYY-WW (ISO week) to YYYYMMDD for the Monday of that week."""
    parts = iso_week_str.split('-')

    year = int(parts[-2])
    week = int(parts[-1])

    # ISO week: Monday is day 1
    monday_date = date.fromisocalendar(year, week, 1)
    return monday_date.strftime('%Y%m%d')


def count_lines(path: str) -> int:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return sum(1 for _ in f)


def read_vertex_file(path: str) -> Set[str]:
    result: Set[str] = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            result.add(line.strip())
    return result


def read_edge_file(path: str, id_to_domain: Dict[int, str]) -> Set[Tuple[str, str]]:
    result: Set[Tuple[str, str]] = set()
    get = id_to_domain.get
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            src_id = int(parts[0])
            dst_id = int(parts[1])
            src = get(src_id)
            dst = get(dst_id)
            if src is None or dst is None:
                continue  # skip if an endpoint wasn't present in vertices map
            result.add((src, dst))
    return result


def open_file(path: str) -> Callable[..., IO]:
    return gzip.open if path.endswith('.gz') else open


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
            else:
                # Global encoder (In our case the RNIEncoder)
                xs.append(encoder(df.shape[0]))

        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    mapping: Dict,
    encoders: Dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    usecols = [src_index_col, dst_index_col]
    if encoders is not None:
        usecols += [col for col in encoders if col not in usecols]

    df = pd.read_csv(path, usecols=usecols)

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
