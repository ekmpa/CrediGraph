import argparse

import torch
from torch.utils.data import DataLoader

from tgrag.gnn.model import Model
from tgrag.utils.logger import setup_logging
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='MLP Text Experiment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--feature-file',
    type=str,
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
    help='Number of hidden channels.',
)
parser.add_argument(
    '--out-channels',
    type=int,
    help='Number of out channels before the final MLP.',
)
parser.add_argument(
    '--num-layers',
    type=int,
    help='Number of Residual FF layers.',
)
parser.add_argument(
    '--dropout',
    type=float,
    help='The dropout.',
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
    help='Seed for reproducibility.',
)


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> None:
    model.train()


def run_ff_experiment() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)
    seed_everything(args.seed)
    mlp = Model(
        model_name='FF',
        normalization=args.normalization,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


if __name__ == '__main__':
    run_ff_experiment()
