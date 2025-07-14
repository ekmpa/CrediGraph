from typing import Dict, Tuple

import pandas as pd
import torch
from torch import Tensor


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
