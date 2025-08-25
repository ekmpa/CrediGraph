import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class RNIEncoder(Encoder):
    def __call__(self, length: int) -> Tensor:
        return torch.rand(length).unsqueeze(1)
