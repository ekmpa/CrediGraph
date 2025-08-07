from typing import Type, Union

import torch
from torch import Tensor, nn

from tgrag.gnn.modules import (
    GATModule,
    GCNModule,
    ResidualModuleWrapper,
    SAGEModule,
)

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class Model(torch.nn.Module):
    modules: dict[str, torch.nn.Module] = {
        'GCN': GCNModule,
        'SAGE': SAGEModule,
        'GAT': GATModule,
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
        normalization_cls = self.normalization_map[normalization]
        # TODO: Here exists the matmul dimension mismatch, the features dim is changed:
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
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                dropout=dropout,
            )
            self.re_modules.append(residual_module)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for re_module in self.re_modules:
            # TODO: The next module will not match dimensions
            x = re_module(x, edge_index)

        x = self.output_normalization(x)
        x = self.output_linear(x)
        x = x.sigmoid()
        return x
