import argparse
import logging
from pathlib import Path
from typing import Dict, cast

import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.gnn.model import Model
from tgrag.utils.args import ModelArguments, parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.plot import (
    plot_pred_target_distributions_histogram,
    plot_regression_scatter_tensor,
)
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Get Predictions for test set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def run_get_test_predictions(
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
    weight_directory: Path,
    target: str,
) -> None:
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    test_idx = dataset.get_idx_split()['test']
    logging.info(f'Length of testing indices: {len(test_idx)}')
    logging.info('Mapping returned.')
    model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info('Model Loaded.')
    model.eval()

    test_targets = dataset[0].y[test_idx]
    logging.info(f'Target values: {test_targets}')
    indices = torch.tensor(test_idx, dtype=torch.long)

    loader = NeighborLoader(
        data,
        input_nodes=indices,
        num_neighbors=[30, 30, 30],
        batch_size=1024,
        shuffle=False,
    )
    logging.info(f'Test indices loader  created for {len(indices)} nodes.')

    num_nodes = data.num_nodes
    all_preds = torch.zeros(num_nodes, 1)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'batch'):
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index)
            seed_nodes = batch.n_id[: batch.batch_size]
            all_preds[seed_nodes] = preds[: batch.batch_size].cpu()

    test_predictions = all_preds[indices]

    abs_errors = (test_predictions - test_targets).abs()

    min_error = abs_errors.min().item()
    max_error = abs_errors.max().item()

    logging.info(f'Min Absolute Error: {min_error:.4f}')
    logging.info(f'Max Absolute Error: {max_error:.4f}')

    plot_pred_target_distributions_histogram(
        preds=test_predictions,
        targets=test_targets,
        model_name=model_arguments.model,
        target=target,
    )
    plot_regression_scatter_tensor(
        preds=test_predictions,
        targets=test_targets,
        model_name=model_arguments.model,
        target=target,
    )


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
    }

    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoder_class = encoder_classes[value]
        encoding_dict[index] = encoder_class

    dataset = TemporalDataset(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
        processed_dir=f'{scratch}/{meta_args.processed_location}',
    )
    logging.info('In-Memory Dataset loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_get_test_predictions(
            experiment_arg.model_args,
            dataset,
            weight_directory,
            target=meta_args.target_col,
        )


if __name__ == '__main__':
    main()
