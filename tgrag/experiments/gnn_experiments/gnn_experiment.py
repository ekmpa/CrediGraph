import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.experiments.gnn_experiments.baseline import (
    evaluate_mean,
    evaluate_rand,
)
from tgrag.gnn.model import Model
from tgrag.utils.args import DataArguments, ModelArguments
from tgrag.utils.logger import Logger
from tgrag.utils.plot import plot_avg_rmse_loss
from tgrag.utils.save import save_loss_results


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
) -> Tuple[float, float]:
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad()
    total_loss = 0
    total_nodes = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        loss = F.mse_loss(preds[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_nodes += 1
        all_preds.append(preds[train_mask])
        all_targets.append(targets[train_mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    mse = total_loss / total_nodes
    return (mse, r2)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
) -> Tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_nodes = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(preds[mask], targets[mask])
        total_loss += loss.item()
        total_nodes += 1
        # total_loss += loss.item() * mask.sum().item()
        # total_nodes += mask.sum().item()
        all_preds.append(preds[mask])
        all_targets.append(targets[mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    mse = total_loss / total_nodes
    return (mse, r2)


def run_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
) -> None:
    is_random = model_arguments.model.upper() == 'RANDOM'
    is_mean = model_arguments.model.upper() == 'MEAN'
    data = dataset[0]
    data.y = data.y.squeeze(1)
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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

    logger = Logger(model_arguments.runs)
    loss_tuple_run: List[List[Tuple[float, float, float]]] = []
    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch: List[Tuple[float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            if not is_random and not is_mean:
                train(model, train_loader, optimizer)
                train_mse = evaluate(model, train_loader, 'train_mask')[0]
                valid_mse = evaluate(model, val_loader, 'valid_mask')[0]
                test_mse = evaluate(model, test_loader, 'test_mask')[0]
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
