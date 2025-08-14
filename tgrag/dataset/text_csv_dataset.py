from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tgrag.encoders.encoder import Encoder


class TextCSVDataset(Dataset):
    def __init__(
        self, csv_path: str, text_col: str, label_col: str, encode_fn: Encoder
    ):
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).to_list()
        labels = df[label_col].to_numpy()

        self.labels = torch.from_numpy(labels).float()

        if hasattr(encode_fn, '__call__'):
            embeds = encode_fn(texts)
        else:
            raise ValueError('encode_fn must be callable.')

        self.x = torch.as_tensor(
            embeds, dtype=torch.float32
        )  # (N, D -> depends on encoder)
        self.y = torch.as_tensor(labels, dtype=torch.float32)  # (N,)

        self.domains = df.get('Domain_Name')

        self.num_features = self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return x, y
