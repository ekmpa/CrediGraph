import logging
import pickle
from typing import Dict, List, Tuple, Type, cast

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
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
    model_name: str,
) -> float:
    model.train()

    optimizer.zero_grad()
    if model_name == 'GAT':
        out = model(data.x, data.edge_index)[train_idx]
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.mse_loss(out.squeeze(), data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


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
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = model_class(
        data.num_features,
        model_arguments.hidden_channels,
        1,
        model_arguments.num_layers,
        model_arguments.dropout,
    ).to(device)

    logger = Logger(model_arguments.runs)

    loss_tuple_run: List[List[Tuple[float, float, float]]] = []
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        logging.info('*** Training ***')
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch: List[Tuple[float, float, float]] = []
        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            loss = train(model, data, train_idx, optimizer, model_arguments.model)
            result = test(model, data, split_idx, model_arguments.model)
            loss_tuple_epoch.append(result)
            logger.add_result(run, result)

            if epoch % model_arguments.log_steps == 0:
                train_loss, valid_loss, test_loss = result
                print(
                    f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train Loss: {100 * train_loss:.2f}%, '
                    f'Valid Loss: {100 * valid_loss:.2f}% '
                    f'Test Loss: {100 * test_loss:.2f}%'
                )
        loss_tuple_run.append(loss_tuple_epoch)

    logger.print_statistics()
    plot_avg_rmse_loss(loss_tuple_run, model_arguments.model)
    save_loss_results(loss_tuple_run, model_arguments.model, model_arguments.encoder)
