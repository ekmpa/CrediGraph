import numpy as np
import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class NormEncoder(Encoder):
    def __call__(self, input: np.ndarray) -> Tensor:
        input = input.astype(np.float32)
        if input.ndim == 1:
            input = input.reshape(-1, 1)

        max = input.max(axis=0, keepdims=True)
        min = input.min(axis=0, keepdims=True)

        normalized = (input - min) / (max - min)
        return torch.tensor(normalized, dtype=torch.float32)
