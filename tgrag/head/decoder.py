import torch
from torch import Tensor
from torch.nn import Linear


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin_node = Linear(in_dim, hidden_dim)
        self.out = Linear(hidden_dim, out_dim)

    def reset_parameters(self) -> None:
        self.lin_node.reset_parameters()
        self.out.reset_parameters()

    def forward(self, node_embedding: Tensor) -> Tensor:
        h = self.lin_node(node_embedding)
        h = h.relu()
        h = self.out(h)
        return h
