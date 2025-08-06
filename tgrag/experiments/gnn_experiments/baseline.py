import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader


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
        random_out = torch.full(batch.y[mask].size(), 0.5).to(device)
        loss = F.binary_cross_entropy(random_out, batch.y[mask])
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
