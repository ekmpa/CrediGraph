import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.experiments.gnn_experiments.baseline import (
    evaluate_fb_mean,
    evaluate_fb_rand,
)
from tgrag.gnn.model import Model
from tgrag.utils.args import DataArguments, ModelArguments
from tgrag.utils.logger import Logger
from tgrag.utils.plot import plot_avg_rmse_loss
from tgrag.utils.save import save_loss_results


def train(
    model: torch.nn.Module,
    data: TemporalDataset,
    train_idx: torch.Tensor,
    optimizer: torch.optim.AdamW,
) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.mse_loss(out.squeeze(), data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: TemporalDataset,
    split_idx: Dict,
) -> Tuple[float, float, float]:
    model.eval()
    out = model(data.x, data.edge_index)

    y_true = data.y
    y_pred = out

    train_rmse = F.mse_loss(
        y_pred[split_idx['train']], y_true[split_idx['train']]
    ).item()
    valid_rmse = F.mse_loss(
        y_pred[split_idx['valid']], y_true[split_idx['valid']]
    ).item()
    test_rmse = F.mse_loss(y_pred[split_idx['test']], y_true[split_idx['test']]).item()

    return train_rmse, valid_rmse, test_rmse


def run_gnn_baseline_full_batch(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
) -> None:
    is_random = model_arguments.model.upper() == 'RANDOM'
    is_mean = model_arguments.model.upper() == 'MEAN'
    logging.info('Running Full-Batch')
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')
    device = torch.device(device)

    data = dataset[0]
    data.y = data.y.squeeze(1)
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

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
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch: List[Tuple[float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            if not is_random and not is_mean:
                train(model, data, train_idx, optimizer)
                result = evaluate(model, data, split_idx)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)
            elif is_random:
                result = evaluate_fb_rand(model, data, split_idx)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)
            else:
                result = evaluate_fb_mean(model, data, split_idx)
                loss_tuple_epoch.append(result)
                logger.add_result(run, result)

        loss_tuple_run.append(loss_tuple_epoch)

    logging.info(logger.get_statistics())
    logging.info('Constructing RMSE plots')
    plot_avg_rmse_loss(loss_tuple_run, model_arguments.model, 'todo')
    logging.info('Saving pkl of results')
    save_loss_results(loss_tuple_run, model_arguments.model, 'todo')
