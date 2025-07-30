import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def train_rand(
    train_loader: NeighborLoader,
) -> float:
    total_loss = 0
    total_nodes = 0
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        loss = F.binary_cross_entropy(torch.rand(batch.y.size(0), 1), batch.y)
        total_loss += loss.item() * batch.y.size(0)
        total_nodes += batch.y.size(0)

    return torch.tensor(total_loss / total_nodes).item()


@torch.no_grad()
def evaluate_rand(
    model: torch.nn.Module,
    loader: NeighborLoader,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        random_out = torch.rand(batch.y.size(0)).to(device)
        loss = F.binary_cross_entropy(random_out, batch.y)
        total_loss += loss.item() * batch.y.size(0)
        total_nodes += batch.y.size(0)

    return torch.tensor(total_loss / total_nodes).item()
