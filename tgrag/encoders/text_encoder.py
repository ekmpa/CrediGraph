import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class TextEncoder(Encoder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str | None = None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: np.ndarray) -> Tensor:
        x = self.model.encode(
            input, show_progress_bar=True, convert_to_tensor=True, device=self.device
        )
        return x.to(self.device)
