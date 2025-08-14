from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from tgrag.encoders.encoder import Encoder


class TextCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_col: str,
        label_col: str,
        encode_fn: Encoder,
        max_seq_length: int = 256,
        batch_size: int = 64,
    ):
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).to_list()
        labels = df[label_col].to_numpy()

        self.labels = torch.from_numpy(labels).float()

        all_embeds = []

        if hasattr(encode_fn, '__call__'):
            embeds = encode_fn(texts)
        else:
            raise ValueError('encode_fn must be callable.')

        for i in tqdm(range(0, len(texts), batch_size), desc='embedding batchs'):
            batch_texts = texts[i : i + batch_size]
            with torch.no_grad():
                embeds = encode_fn(batch_texts)  # (B, D)
            all_embeds.append(embeds.cpu())

        self.x = torch.cat(all_embeds, dim=0)  # (N, D -> depends on encoder)
        self.y = torch.as_tensor(labels, dtype=torch.float32)  # (N,)

        self.domains = df.get('Domain_Name')

        self.num_features = self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return x, y
