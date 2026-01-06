from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from tgrag.encoders.encoder import Encoder


class TextCSVDataset(Dataset):
    """Dataset that encodes text from a CSV file into fixed embeddings with labels."""

    def __init__(
        self,
        csv_path: str,
        text_col: str,
        label_col: str,
        encode_fn: Encoder,
        batch_size: int = 64,
    ):
        """Load a CSV file, encode text, and construct embedding/label tensors.

        Parameters:
            csv_path : str
                Path to the input CSV file.
            text_col : str
                Name of the column containing text to encode.
            label_col : str
                Name of the column containing target labels.
            encode_fn : Encoder
                Callable that maps a list of strings to a tensor of embeddings.
            batch_size : int, optional
                Number of texts to encode per batch (default: 64).

        Raises:
            ValueError
                If encode_fn is not callable.
        """
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).to_list()
        labels = df[label_col].to_numpy()

        self.labels = torch.from_numpy(labels).float()

        all_embeds = []

        if hasattr(encode_fn, '__call__'):
            for i in tqdm(range(0, len(texts), batch_size), desc='embedding batchs'):
                batch_texts = texts[i : i + batch_size]
                with torch.no_grad():
                    embeds = encode_fn(batch_texts)  # (B, D)
                all_embeds.append(embeds.cpu())
        else:
            raise ValueError('encode_fn must be callable.')

        self.x = torch.cat(all_embeds, dim=0)  # (N, D -> depends on encoder)
        self.y = torch.as_tensor(labels, dtype=torch.float32)  # (N,)

        self.domains = df.get('Domain_Name')

        self.num_features = self.x.shape[1]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return the (embedding, label) pair for a given index.

        Parameters:
            idx : int
                Sample index.

        Returns:
            Tuple[Tensor, Tensor]
                (x, y) where x is the embedding tensor and y is the label.
        """
        x = self.x[idx]
        y = self.y[idx]
        return x, y
