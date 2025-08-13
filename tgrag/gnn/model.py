from typing import Type, Union

import torch
from torch import Tensor, nn

from tgrag.gnn.modules import (
    FFModule,
    GATModule,
    GCNModule,
    NodePredictor,
    ResidualModuleWrapper,
    SAGEModule,
)

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class Model(torch.nn.Module):
    modules: dict[str, torch.nn.Module] = {
        'GCN': GCNModule,
        'SAGE': SAGEModule,
        'GAT': GATModule,
        'FF': FFModule,
    }
    normalization_map: dict[str, NormalizationType] = {
        'none': torch.nn.Identity,
        'LayerNorm': torch.nn.LayerNorm,
        'BatchNorm': torch.nn.BatchNorm1d,
    }

    def __init__(
        self,
        model_name: str,
        normalization: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.model_name = model_name
        normalization_cls = self.normalization_map[normalization]
        self.input_linear = nn.Linear(
            in_features=in_channels, out_features=hidden_channels
        )
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.re_modules = nn.ModuleList()

        for _ in range(num_layers):
            residual_module = ResidualModuleWrapper(
                module=self.modules[model_name],
                normalization=normalization_cls,
                dim=hidden_channels,
                dropout=dropout,
            )
            self.re_modules.append(residual_module)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        self.node_predictor = NodePredictor(in_dim=out_channels, out_dim=1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for re_module in self.re_modules:
            x = re_module(x, edge_index)

        x = self.output_normalization(x)
        x = self.output_linear(x)
        x = self.node_predictor(x)
        return x
