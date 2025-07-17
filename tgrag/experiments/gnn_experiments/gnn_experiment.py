import logging
import pickle
from typing import Dict, List, Tuple, Type, cast

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.log_encoding import LogEncoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.gnn.GAT import GAT
from tgrag.gnn.gCon import GCN
from tgrag.gnn.SAGE import SAGE
from tgrag.utils.args import DataArguments, ModelArguments
from tgrag.utils.logger import Logger
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import plot_avg_rmse_loss

MODEL_CLASSES: Dict[str, Type[torch.nn.Module]] = {
    'GCN': GCN,
    'GAT': GAT,
    'SAGE': SAGE,
}

ENCODER_CLASSES: Dict[str, Encoder] = {
    'RNI': RNIEncoder(),
    'LOG': LogEncoder(),
}


def save_loss_results(
    loss_tuple_run: List[List[Tuple[float, float, float]]],
    model_name: str,
    encoder_name: str,
) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'logs' / model_name / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'loss_tuple_run.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(loss_tuple_run, f)


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Adam,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    optimizer.zero_grad()
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.mse_loss(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(
    model: torch.nn.Module,
    data: TemporalDataset,
    split_idx: Dict,
    model_name: str,
) -> Tuple[float, float, float]:
    model.eval()
    if model_name == 'GAT':
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.adj_t)

    y_true = data.y
    y_pred = out

    train_rmse = torch.sqrt(
        F.mse_loss(y_pred[split_idx['train']], y_true[split_idx['train']])
    ).item()
    valid_rmse = torch.sqrt(
        F.mse_loss(y_pred[split_idx['valid']], y_true[split_idx['valid']])
    ).item()
    test_rmse = torch.sqrt(
        F.mse_loss(y_pred[split_idx['test']], y_true[split_idx['test']])
    ).item()

    return train_rmse, valid_rmse, test_rmse


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.mse_loss(out.squeeze(), batch.y)
        total_loss += loss.item() * batch.y.size(0)
        total_nodes += batch.y.size(0)

    return torch.sqrt(torch.tensor(total_loss / total_nodes)).item()


def run_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
) -> None:
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    root_dir = get_root_dir()

    model_class = MODEL_CLASSES[model_arguments.model]
    encoder_class = ENCODER_CLASSES[model_arguments.encoder]
    logging.info(
        'Encoder: %s is used on column: %s',
        model_arguments.encoder,
        model_arguments.encoder_col,
    )

    encoding_dict: Dict[str, Encoder] = {model_arguments.encoder_col: encoder_class}

    dataset = TemporalDataset(
        root=f'{root_dir}/data/crawl-data/temporal',
        node_file=cast(str, data_arguments.node_file),
        edge_file=cast(str, data_arguments.edge_file),
        encoding=encoding_dict,
    )
    data = dataset[0]
    data.y = data.y.squeeze(1)
    split_idx = dataset.get_idx_split()

    logging.info(f"Training set size: {split_idx['train'].size()}")
    logging.info(f"Validation set size: {split_idx['valid'].size()}")
    logging.info(f"Testing set size: {split_idx['test'].size()}")

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=[5, 5],
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=[5, 5],
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=[5, 5],
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = model_class(
        data.num_features,
        model_arguments.hidden_channels,
        1,
        model_arguments.num_layers,
        model_arguments.dropout,
        cached=False,
    ).to(device)

    logger = Logger(model_arguments.runs)

    loss_tuple_run: List[List[Tuple[float, float, float]]] = []
    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch: List[Tuple[float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            train(model, train_loader, optimizer)
            train_rmse = evaluate(model, train_loader)
            valid_rmse = evaluate(model, val_loader)
            test_rmse = evaluate(model, test_loader)
            result = (train_rmse, valid_rmse, test_rmse)
            loss_tuple_epoch.append(result)
            logger.add_result(run, result)

        loss_tuple_run.append(loss_tuple_epoch)

    logging.info(logger.get_statistics())
    plot_avg_rmse_loss(loss_tuple_run, model_arguments.model, model_arguments.encoder)
    save_loss_results(loss_tuple_run, model_arguments.model, model_arguments.encoder)
