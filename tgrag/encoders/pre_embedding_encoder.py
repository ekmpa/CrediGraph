import logging
from typing import Dict

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from tgrag.encoders.encoder import Encoder
from tgrag.utils.domain_handler import reverse_domain


class TextEmbeddingEncoder(Encoder):
    def __init__(self, default_dimension: int):
        self.default_dimension = default_dimension

    def __call__(
        self, domain_names: pd.Series, embeddings_lookup: Dict[str, torch.Tensor]
    ) -> Tensor:
        text_embeddings_used = 0
        n = len(domain_names)
        out = torch.empty(
            (n, self.default_dimension), dtype=torch.float32, device='cpu'
        )
        for i, domain_name in tqdm(enumerate(domain_names), desc='Domain lookup'):
            rev_domain_name = reverse_domain(domain_name)
            if rev_domain_name in embeddings_lookup:
                out[i] = embeddings_lookup[rev_domain_name]
                text_embeddings_used += 1
            else:
                out[i] = torch.rand(self.default_dimension, dtype=torch.float32)
        logging.info(f'Dimension of stacked embeddings: {out.shape}')
        logging.info(f'Text embeddings used: {text_embeddings_used}')

        return out  # [num_domains, embedding_dim]
