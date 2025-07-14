import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class LogEncoder(Encoder):
    def __init__(self, scale: float | None = None):
        self.scale = scale

    def __call__(self, input: float) -> Tensor:
        return torch.Tensor(input).unsqueeze(1)
