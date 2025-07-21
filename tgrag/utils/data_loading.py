import glob
import gzip
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import Tensor

from tgrag.utils.matching import reverse_domain
from tgrag.utils.path import get_root_dir


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


def get_ids_from_set(wanted_domains: set[str], source_base: str) -> set[str]:
    """Get the node ids in the graph of the nodes in the wanted_domains set."""
    source_dir = os.path.join(source_base, 'vertices')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return set()

    matched_ids = set()
    for source_file in matches:
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                vertex_id = parts[0].strip()
                rev_domain = parts[1].strip()

                normal_domain = reverse_domain(rev_domain).lower()

                # Match exact or parent domains
                domain_parts = normal_domain.split('.')
                for i in range(len(domain_parts) - 1):
                    candidate = '.'.join(
                        domain_parts[i:]
                    )  # e.g. sub.domain.com, domain.com, com
                    if candidate in wanted_domains:
                        matched_ids.add(vertex_id)
                        break

    # print(f'[INFO] Found {len(matched_ids)} matching vertex IDs.')
    return matched_ids


def get_baseline_domains() -> set[str]:
    """Get a set of baseline domains from cc-baseline-domains.txt."""
    path = os.path.join(get_root_dir(), 'data', 'cc-baseline-domains.txt')
    baseline_domains = set()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            domain = line.strip()
            if domain:  # skip empty lines
                baseline_domains.add(domain)

    print(f'[INFO] Found {len(baseline_domains)} baseline domains')
    return baseline_domains
