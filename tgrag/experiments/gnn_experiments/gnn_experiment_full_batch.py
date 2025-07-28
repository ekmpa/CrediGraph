import logging
import pickle
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
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

ENCODER_MAPPING: Dict[str, int] = {
    'random': 0,
    'pr_val': 1,
    'hc_val': 2,
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
    data: TemporalDataset,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Adam,
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
    logging.info('Running Full-Batch')
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')
    device = torch.device(device)

    model_class = MODEL_CLASSES[model_arguments.model]

    data = dataset[0]
    data.y = data.y.squeeze(1)
    data.x = data.x[:, ENCODER_MAPPING[data_arguments.initial_encoding_col]].unsqueeze(
        -1
    )
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

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
            train(model, data, train_idx, optimizer)
            result = evaluate(model, data, split_idx)
            loss_tuple_epoch.append(result)
            logger.add_result(run, result)

        loss_tuple_run.append(loss_tuple_epoch)

    logging.info(logger.get_statistics())
    logging.info('Constructing RMSE plots')
    plot_avg_rmse_loss(loss_tuple_run, model_arguments.model, 'todo')
    logging.info('Saving pkl of results')
    save_loss_results(loss_tuple_run, model_arguments.model, 'todo')
