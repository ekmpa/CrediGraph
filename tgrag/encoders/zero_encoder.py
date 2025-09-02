import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class ZeroEncoder(Encoder):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def __call__(self, length: int) -> Tensor:
        return torch.zeros(length, self.dimension)
