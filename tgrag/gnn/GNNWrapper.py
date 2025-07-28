import torch
from torch import Tensor

from tgrag.head.decoder import NodePredictor


class GNNWrapper(torch.nn.Module):
    def __init__(self, gnn: torch.nn.Module, predictor: NodePredictor):
        super().__init__()
        self.model = torch.nn.ModuleList([gnn, predictor])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.model[0](x, edge_index)
        x = self.model[1](x)
        return x

    def reset_parameters(self) -> None:
        for module in self.model:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
