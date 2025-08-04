import logging
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.experiments.gnn_experiments.random_baseline import (
    evaluate_mean,
    evaluate_rand,
)
from tgrag.gnn.GAT import GAT
from tgrag.gnn.gCon import GCN
from tgrag.gnn.GNNWrapper import GNNWrapper
from tgrag.gnn.SAGE import SAGE
from tgrag.head.decoder import NodePredictor
from tgrag.utils.args import DataArguments, ModelArguments
from tgrag.utils.logger import Logger
from tgrag.utils.plot import plot_avg_rmse_loss
from tgrag.utils.save import save_loss_results

MODEL_CLASSES: Dict[str, Type[torch.nn.Module]] = {
    'GCN': GCN,
    'GAT': GAT,
    'SAGE': SAGE,
    'RANDOM': GCN,
}

ENCODER_MAPPING: Dict[str, int] = {
    'random': 0,
    'pr_val': 1,
    'hc_val': 2,
}


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Adam,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    optimizer.zero_grad()
    total_nodes = 0
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        # TODO: How many are negative in the output per batch?
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        loss = F.binary_cross_entropy_with_logits(
            out.squeeze()[train_mask], batch.y[train_mask]
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * train_mask.sum().item()
        total_nodes += train_mask.sum().item()

    return total_loss / total_nodes


@torch.no_grad()
def evaluate(
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
        out = model(batch.x, batch.edge_index)
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out.squeeze()[mask], batch.y[mask])
        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    return total_loss / total_nodes


def run_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
) -> None:
    is_random = model_arguments.model.upper() == 'RANDOM'
    is_mean = model_arguments.model.upper() == 'MEAN'
    data = dataset[0]
    data.y = data.y.squeeze(1)
    # data.x = data.x[:, ENCODER_MAPPING[data_arguments.initial_encoding_col]].unsqueeze(
    #     -1
    # )
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model_class = MODEL_CLASSES[model_arguments.model]
    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Train loader created')

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    logging.info('Valid loader created')
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Test loader created')
    gnn = model_class(
        data.num_features,
        model_arguments.hidden_channels,
        model_arguments.embedding_dimension,
        model_arguments.num_layers,
        model_arguments.dropout,
        cached=False,
    ).to(device)
    node_predictor = NodePredictor(model_arguments.embedding_dimension, 5, 1).to(device)
    model = GNNWrapper(gnn, node_predictor)
    logger = Logger(model_arguments.runs)

    loss_tuple_run: List[List[Tuple[float, float, float]]] = []
    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch: List[Tuple[float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            if not is_random and not is_mean:
                train(model, train_loader, optimizer)
                train_mse = evaluate(model, train_loader, 'train_mask')
                valid_mse = evaluate(model, val_loader, 'valid_mask')
                test_mse = evaluate(model, test_loader, 'test_mask')
                result = (train_mse, valid_mse, test_mse)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)
            elif is_random:
                train_mse = evaluate_rand(model, train_loader, 'train_mask')
                valid_mse = evaluate_rand(model, val_loader, 'valid_mask')
                test_mse = evaluate_rand(model, test_loader, 'test_mask')
                result = (train_mse, valid_mse, test_mse)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)
            else:
                train_mse = evaluate_mean(model, train_loader, 'train_mask')
                valid_mse = evaluate_mean(model, val_loader, 'valid_mask')
                test_mse = evaluate_mean(model, test_loader, 'test_mask')
                result = (train_mse, valid_mse, test_mse)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)

        loss_tuple_run.append(loss_tuple_epoch)

    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics())
    logging.info(logger.get_avg_statistics())
    logging.info('Constructing RMSE plots')
    plot_avg_rmse_loss(loss_tuple_run, model_arguments.model, 'TODO')
    logging.info('Saving pkl of results')
    save_loss_results(loss_tuple_run, model_arguments.model, 'TODO')
