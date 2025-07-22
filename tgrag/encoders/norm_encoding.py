import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class NormEncoder(Encoder):
    def __init__(self, scale: float | None = None):
        self.scale = scale

    def __call__(self, input: float) -> Tensor:
        x = torch.as_tensor(input, dtype=torch.float32).unsqueeze(-1)
        x = torch.log1p(x)
        if self.scale:
            x = x * self.scale
        return x
