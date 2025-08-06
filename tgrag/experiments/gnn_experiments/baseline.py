from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from tgrag.dataset.temporal_dataset import TemporalDataset


@torch.no_grad()
def evaluate_mean(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        mean_out = torch.full(batch.y[mask].size(), 0.5).to(device)
        loss = F.binary_cross_entropy(mean_out, batch.y[mask])
        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    return total_loss / total_nodes


@torch.no_grad()
def evaluate_rand(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        random_out = torch.rand(batch.y[mask].size(0)).to(device)
        loss = F.binary_cross_entropy(random_out, batch.y[mask])
        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    return total_loss / total_nodes


@torch.no_grad()
def evaluate_fb_rand(
    model: torch.nn.Module,
    data: TemporalDataset,
    split_idx: Dict,
) -> Tuple[float, float, float]:
    model.eval()

    y_true = data.y

    pred_rand_train = torch.rand(split_idx['train'].size(0))
    pred_rand_valid = torch.rand(split_idx['valid'].size(0))
    pred_rand_test = torch.rand(split_idx['test'].size(0))

    train_rmse = F.mse_loss(pred_rand_train, y_true[split_idx['train']]).item()
    valid_rmse = F.mse_loss(pred_rand_valid, y_true[split_idx['valid']]).item()
    test_rmse = F.mse_loss(pred_rand_test, y_true[split_idx['test']]).item()

    return train_rmse, valid_rmse, test_rmse


@torch.no_grad()
def evaluate_fb_mean(
    model: torch.nn.Module,
    data: TemporalDataset,
    split_idx: Dict,
) -> Tuple[float, float, float]:
    model.eval()

    y_true = data.y

    pred_mean_train = torch.full(split_idx['train'].size(0), 0.5)
    pred_mean_valid = torch.full(split_idx['valid'].size(0), 0.5)
    pred_mean_test = torch.full(split_idx['test'].size(0), 0.5)

    train_rmse = F.mse_loss(pred_mean_train, y_true[split_idx['train']]).item()
    valid_rmse = F.mse_loss(pred_mean_valid, y_true[split_idx['valid']]).item()
    test_rmse = F.mse_loss(pred_mean_test, y_true[split_idx['test']]).item()

    return train_rmse, valid_rmse, test_rmse
