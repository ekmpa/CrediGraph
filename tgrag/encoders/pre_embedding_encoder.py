import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class TextEmbeddingEncoder(Encoder):
    def __init__(self, default_dimension: int):
        self.default_dimension = default_dimension

    def __call__(
        self, domain_names: pd.Series, embeddings_lookup: Dict[str, np.ndarray]
    ) -> Tensor:
        embeddings: List[torch.Tensor] = []
        text_embeddings_used = 0
        for domain_name in domain_names:
            if domain_name in embeddings_lookup:
                vec = torch.as_tensor(
                    embeddings_lookup[domain_name], dtype=torch.float32
                )
                text_embeddings_used += 1
            else:
                vec = torch.rand(self.default_dimension, dtype=torch.float32)
            embeddings.append(vec)
        t = torch.stack(embeddings)
        logging.info(f'Dimension of stacked embeddings: {t.shape}')

        return t  # [num_domains, embedding_dim]
