import numpy as np
import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class NormEncoder(Encoder):
    def __call__(self, input: np.ndarray) -> Tensor:
        input = input.astype(np.float32)
        if input.ndim == 1:
            input = input.reshape(-1, 1)

        mean = input.mean(axis=0, keepdims=True)
        std = input.std(axis=0, keepdims=True) + 1e-8

        normalized = (input - mean) / std
        return torch.tensor(normalized, dtype=torch.float32)
