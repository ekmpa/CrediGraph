from typing import Type, Union

from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class ResidualModuleWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        normalization: NormalizationType,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.normalization = normalization(hidden_channels)
        self.module = module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_res = self.normalization(x)
        x_res = self.module(x, edge_index)
        x = x + x_res
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_channels, out_features=in_channels)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(
            in_channels=in_channels, hidden_channels=hidden_channels, dropout=dropout
        )
        self.convs = GCNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.convs(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class SAGEModule(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(
            in_channels=in_channels, hidden_channels=hidden_channels, dropout=dropout
        )
        self.convs = SAGEConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.convs(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class GATModule(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(
            in_channels=in_channels, hidden_channels=hidden_channels, dropout=dropout
        )
        self.convs = GATConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.convs(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x
