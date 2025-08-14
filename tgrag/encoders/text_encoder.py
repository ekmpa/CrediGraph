import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class TextEncoder(Encoder):
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
        device: str | None = None,
        batch_size: int = 8,
        use_fp16: bool = True,
    ):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

        if use_fp16 and self.device is not None and 'cuda' in self.device:
            self.model = self.model.half()

        self.batch_size = batch_size

    def __call__(self, input: np.ndarray) -> Tensor:
        x = self.model.encode(
            input,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()
