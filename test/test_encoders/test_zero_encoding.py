import torch

from tgrag.encoders.zero_encoder import ZeroEncoder


def test_zero_call():
    z = ZeroEncoder(4)
    x = z(5)
    assert isinstance(x, torch.Tensor)
    print(x)
    print(x.shape)
    assert x.shape == torch.Size([5, 4])
