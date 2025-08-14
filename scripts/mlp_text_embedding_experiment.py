import argparse
import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tgrag.dataset.text_csv_dataset import TextCSVDataset
from tgrag.encoders.text_encoder import TextEncoder
from tgrag.gnn.model import Model
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import Scoring, plot_avg_loss
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='MLP Text Experiment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--feature-file',
    type=str,
    default='data/dqr/merged_domains_text_pc1.csv',
    help='Path to 11k dqr domains with text content in CSV format',
)
parser.add_argument(
    '--normalization',
    type=str,
    default='LayerNorm',
    help='Normalization type',
    choices=['none', 'LayerNorm', 'BatchNorm'],
)
parser.add_argument(
    '--hidden-channels',
    type=int,
    default=256,
    help='Number of hidden channels.',
)
parser.add_argument(
    '--out-channels',
    type=int,
    default=64,
    help='Number of out channels before the final MLP.',
)
parser.add_argument(
    '--num-layers',
    type=int,
    default=3,
    help='Number of Residual FF layers.',
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.1,
    help='The dropout.',
)
parser.add_argument(
    '--device',
    type=int,
    default=1,
    help='The cuda device.',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='The number of epochs.',
)
parser.add_argument(
    '--runs',
    type=int,
    default=1,
    help='The number of trials.',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='MLP_text_experiment.log',
    help='Name of log file at project root.',
)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Seed for reproducibility.',
)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    total_batches = 0
    for batch in tqdm(train_loader, desc='Batchs'):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(x=batch.x)
        targets = batch.y
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1

    mse = total_loss / total_batches
    return mse


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0
    total_batches = 0
    for batch in loader:
        batch.to(device)
        preds = model(x=batch.x)
        targets = batch.y
        loss = F.mse_loss(preds, targets)
        total_loss += loss.item()
        total_batches += 1

    mse = total_loss / total_batches
    return mse


def run_ff_experiment() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)
    seed_everything(args.seed)
    root = get_root_dir()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 6,
    }

    encoder = TextEncoder()
    data = TextCSVDataset(
        csv_path=root / args.feature_file,
        text_col='wet_record_txt',
        label_col='pc1',
        encode_fn=encoder,
    )
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(test_dataset, **params)

    mlp = Model(
        model_name='FF',
        normalization=args.normalization,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr)
    loss_run_mse: List[List[Tuple[float, float, float]]] = []
    logging.info('*** Training ***')
    for run in tqdm(range(args.runs), desc='Runs'):
        loss_epoch_mse: List[Tuple[float, float, float]] = []
        for _ in tqdm(range(1, 1 + args.epochs), desc='Epochs'):
            train_mse = train(
                model=mlp, train_loader=train_loader, optimizer=optimizer, device=device
            )
            test_mse = evaluate(model=mlp, loader=test_loader, device=device)
            val_mse = 0.0
            result = (train_mse, val_mse, test_mse)
            loss_epoch_mse.append(result)
        loss_run_mse.append(loss_epoch_mse)
    logging.info('*** Constructing Plots ***')
    plot_avg_loss(loss_run_mse, mlp.model_name, Scoring.mse, 'mse_loss_plot.png')


if __name__ == '__main__':
    run_ff_experiment()
