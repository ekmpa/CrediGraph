# Reader helpers
# Incl. readers for graphs at various steps and label datasets.

import csv
import gzip
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Set, TextIO, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from tgrag.utils.domain_handler import normalize_domain
from tgrag.utils.path import get_root_dir

# For graphs
# ----------


def line_reader(path: str | Path) -> Iterator[str]:
    """Yield lines from a text or gzip file without trailing newlines.

    Parameters:
        path : str or pathlib.Path
            Path to a text file or gzip-compressed file.

    Yields:
        str
            Each line from the file with the trailing newline removed.
    """
    p = Path(path)

    opener: Callable[..., TextIO]
    if p.suffix == '.gz':
        opener = gzip.open
    else:
        opener = open

    with opener(p, 'rt', encoding='utf-8', newline='') as f:
        for line in f:
            yield line.rstrip('\n')


def read_vertex_file(path: str) -> Set[str]:
    """Read a text or gzip vertex file into a set of strings.

    Parameters:
        path : str
            Path to the gzip-compressed vertex file.

    Returns:
        set of str
            All unique vertex identifiers found in the file.
    """
    result: Set[str] = set()
    for line in line_reader(path):
        if line:
            result.add(line.strip())
    return result


def read_edge_file(path: str, id_to_domain: Dict[int, str]) -> Set[Tuple[str, str]]:
    """Read a text or gzip edge file and map numeric IDs to domain strings.

    Parameters:
        path : str
            Path to the gzip-compressed edge file.
        id_to_domain : dict[int, str]
            Mapping from numeric node IDs to domain names.

    Returns:
        set of (str, str)
            Set of (source_domain, destination_domain) tuples.
    """
    result: Set[Tuple[str, str]] = set()
    get = id_to_domain.get
    for line in line_reader(path):
        if not line:
            continue
        parts = line.strip().split('\t')
        src_id = int(parts[0])
        dst_id = int(parts[1])
        src = get(src_id)
        dst = get(dst_id)
        if src is None or dst is None:
            continue  # skip if an endpoint wasn't present in vertices map
        result.add((src, dst))
    return result


def load_edges(path: str) -> List[Tuple[str, str]]:
    """Load an edge list from a text or gzip file, expected to have (source, domain) per line.

    Parameters:
        path : str
            Path to the edge file.

    Returns:
        list of (str, str)
            List of edge tuples.
    """
    edges = []
    for line in tqdm(line_reader(path), desc='Loading edge file'):
        parts = line.strip().split()
        if len(parts) == 2:
            edges.append((parts[0], parts[1]))
    return edges


def load_node_domain_map(path: str | Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load a node-to-domain and domain-to-node mapping from a text or gzip file.

    Parameters:
        path : str or pathlib.Path
            Path to the node file, expected to have (node_id, domain) per line.

    Returns:
        (dict, dict)
            A tuple (id_to_domain, domain_to_id).
    """
    id_to_domain: Dict[str, str] = {}
    domain_to_id: Dict[str, str] = {}

    for line in tqdm(line_reader(path), desc='Loading node file'):
        parts = line.strip().split()
        if len(parts) == 2:
            node_id, domain = parts
            id_to_domain[node_id] = domain
            domain_to_id[domain] = node_id

    return id_to_domain, domain_to_id


def _encode_columns(df: pd.DataFrame, encoders: Dict) -> torch.Tensor:
    """Encode dataframe columns using provided encoders and concatenate results.

    Parameters:
        df : pandas.DataFrame
            Input dataframe containing columns to encode.
        encoders : dict
            Mapping from column name to encoder function.

    Returns:
        torch.Tensor
            Concatenated tensor of encoded column values.
    """
    xs = []
    for key, encoder in encoders.items():
        if key in df.columns:
            xs.append(encoder(df[key].values))
        else:
            xs.append(encoder(len(df)))
    return torch.cat(xs, dim=-1)


def load_node_csv(
    path: str,
    index_col: int,
    encoders: Dict | None = None,
    chunk_size: int = 500_000,
) -> Tuple[torch.Tensor | None, Dict, pd.Index]:
    """Load a node CSV file, build an index mapping, and optionally encode features.

    Parameters:
        path : str
            Path to the node CSV file.
        index_col : int
            Column index to use as node identifier.
        encoders : dict or None, optional
            Mapping from column names to encoder functions.
        chunk_size : int, optional
            Number of rows to read per chunk.

    Returns:
        (torch.Tensor or None, dict, pandas.Index)
            Feature tensor (or None), mapping from node id to index, and row index.
    """
    dfs = []

    with pd.read_csv(path, index_col=index_col, chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc='Reading node CSV'):
            dfs.append(chunk)

    df = pd.concat(dfs, axis=0)

    mapping = {idx: i for i, idx in enumerate(df.index.unique())}

    x = None
    if encoders:
        xs = []
        for key, encoder in encoders.items():
            if key in df.columns:
                xs.append(encoder(df[key].values))
            elif key == 'pre':
                xs.append(encoder(df['domain'], get_seed_embeddings()))
            else:
                xs.append(encoder(len(df)))

        x = xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)

    return x, mapping, pd.RangeIndex(len(df))


def load_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    mapping: Dict,
    encoders: Dict | None = None,
    chunk_size: int = 500_000,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Load an edge CSV file and construct edge indices and optional attributes.

    Parameters:
        path : str
            Path to the edge CSV file.
        src_index_col : str
            Column name for source node identifiers.
        dst_index_col : str
            Column name for destination node identifiers.
        mapping : dict
            Mapping from node identifiers to integer indices.
        encoders : dict or None, optional
            Mapping from column names to encoder functions.
        chunk_size : int, optional
            Number of rows to read per chunk.

    Returns:
        (torch.Tensor, torch.Tensor or None)
            Edge index tensor and optional edge attribute tensor.
    """
    usecols = [src_index_col, dst_index_col]
    if encoders:
        usecols += [c for c in encoders if c not in usecols]

    dfs = []
    with pd.read_csv(path, usecols=usecols, chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc='Reading edge CSV'):
            dfs.append(chunk)

    df = pd.concat(dfs, axis=0)

    src = torch.tensor([mapping[s] for s in df[src_index_col]], dtype=torch.long)
    dst = torch.tensor([mapping[d] for d in df[dst_index_col]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr = None
    if encoders:
        edge_attr = _encode_columns(df, encoders)

    return edge_index, edge_attr


def load_large_edge_csv(
    path: str,
    src_index_col: str,
    dst_index_col: str,
    mapping: Dict,
    encoders: Dict | None = None,
    chunk_size: int = 500_000,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Load a large edge CSV file in chunks and construct edge tensors.

    Parameters:
        path : str
            Path to the edge CSV file.
        src_index_col : str
            Column name for source node identifiers.
        dst_index_col : str
            Column name for dest node identifiers.
        mapping : dict
            Mapping from node identifiers to integer indices.
        encoders : dict or None, optional
            Mapping from column names to encoder functions.
        chunk_size : int, optional
            Number of rows to read per chunk.

    Returns:
        (torch.Tensor, torch.Tensor or None)
            Edge index tensor and optional edge attribute tensor.
    """
    usecols = [src_index_col, dst_index_col]
    if encoders:
        usecols += [c for c in encoders if c not in usecols]

    src_chunks, dst_chunks, attr_chunks = [], [], []

    with pd.read_csv(path, usecols=usecols, chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc='Reading edge CSV'):
            src_chunks.append(
                torch.tensor(
                    [mapping[s] for s in chunk[src_index_col]], dtype=torch.long
                )
            )
            dst_chunks.append(
                torch.tensor(
                    [mapping[d] for d in chunk[dst_index_col]], dtype=torch.long
                )
            )

            if encoders:
                attr_chunks.append(_encode_columns(chunk, encoders))

    src = torch.cat(src_chunks)
    dst = torch.cat(dst_chunks)
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr = torch.cat(attr_chunks) if attr_chunks else None

    return edge_index, edge_attr


def get_seed_embeddings(
    file_name: str = 'data/dqr/labeled_11k_domainname_emb/labeled_11k_domainName_emb.pkl',
) -> Dict[str, torch.Tensor]:
    """Load precomputed domain embeddings.

    Parameters:
        None

    Returns:
        dict[str, torch.Tensor]
            Mapping from domain string to embedding tensor.
    """
    root = get_root_dir()
    path = Path(root) / file_name

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


# For labels
# ----------


def _csv_rows(path: Path) -> Iterable[dict[str, str]]:
    """Iterate over rows of a CSV file as dictionaries.

    Parameters:
        path : pathlib.Path
            Path to a CSV file.

    Yields:
        dict[str, str]
            Row mapping column names to values.
    """
    with path.open('r', encoding='utf-8') as f:
        yield from csv.DictReader(f)


def get_full_dict() -> Dict[str, List[float]]:
    """Load domain rating metrics from the DQR dataset.

    Parameters:
        None

    Returns:
        dict[str, list[float]]
            Mapping from domain string to a list of numeric metric values.
    """
    path = Path(get_root_dir()) / 'data' / 'dqr' / 'domain_ratings.csv'
    result: Dict[str, List[float]] = {}

    with path.open('r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            result[parts[0]] = [float(x) for x in parts[1:]]

    return result


def load_domains(path: Path, domain_col: str = 'domain') -> set[str]:
    """Load and normalize domain names from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to CSV file containing domain values.
        domain_col : str, optional
            Name of the column containing domain strings.

    Returns:
        set[str]
            Set of normalized domain strings.
    """
    return {
        d for row in _csv_rows(path) if (d := normalize_domain(row.get(domain_col)))
    }


def read_weak_labels(path: Path) -> dict[str, int]:
    """Read weak (binary) labels from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to a CSV file with columns "domain" and "label".

    Returns:
        dict[str, int]
            Mapping from domain to integer label.
    """
    result: dict[str, int] = {}

    for row in _csv_rows(path):
        d = normalize_domain(row.get('domain'))
        l = row.get('label')
        if d and l is not None:
            result[d] = int(l)

    return result


def read_reg_scores(path: Path, score_col: str = 'pc1') -> dict[str, float]:
    """Read regression scores from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to a CSV file containing domain scores.
        score_col : str, optional
            Name of the column containing the regression score.

    Returns:
        dict[str, float]
            Mapping from domain to floating-point regression score.
    """
    result: dict[str, float] = {}

    for row in _csv_rows(path):
        d = normalize_domain(row.get('domain'))
        s = row.get(score_col)
        if d and s is not None:
            try:
                result[d] = float(s)
            except ValueError:
                continue

    return result


def collect_merged(paths: Iterable[Path], output_csv: Path) -> dict[str, list[float]]:
    """Collect and aggregate domain labels from multiple CSV files that have labelled domains.

    Parameters:
        paths : iterable of pathlib.Path
            Input CSV file paths containing at least columns "domain" and "label".
        output_csv : pathlib.Path
            Path to the output CSV file (excluded from reading).

    Returns:
        dict[str, list[float]]
            Mapping from domain string to a list of numeric label values.
    """
    domain_labels: dict[str, list[float]] = defaultdict(list)

    for path in paths:
        if path.name == output_csv.name:
            continue

        for row in _csv_rows(path):
            d = normalize_domain(row.get('domain'))
            l = row.get('label')
            if d and l is not None:
                try:
                    domain_labels[d].append(float(l))
                except ValueError:
                    continue

    return domain_labels
