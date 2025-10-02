import argparse
import logging
from pathlib import Path
from typing import Dict, cast

import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.gnn.model import Model
from tgrag.utils.args import ModelArguments, parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.matching import reverse_domain
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Weak Supervision Experiment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def run_weak_supervision_forward(
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
    weight_directory: Path,
) -> None:
    root = get_root_dir()
    phishing_dict: Dict[str, str] = {
        'IP2Location': 'data/phishing_data/cc_dec_2024_PhishDataset_legit_domains.csv',
        'URLHaus': 'data/phishing_data/cc_dec_2024_URLhaus_domains.csv',
        'PhishTank': 'data/phishing_data/cc_dec_2024_phishtank_domains.csv',
    }
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}'
    mapping = dataset.get_mapping()
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
    model.eval()
    for dataset_name, path in phishing_dict.items():
        logging.info(f'Predictions of {dataset_name}')
        df = pd.read_csv(root / path)
        phishing_indices = [
            mapping.get(reverse_domain(domain.strip())) for domain in df['domain']
        ]
        phishing_indices = [i for i in phishing_indices if i is not None]
        phishing_indices = torch.tensor(phishing_indices, dtype=torch.long)

        phishing_loader = NeighborLoader(
            data,
            input_nodes=phishing_indices,
            num_neighbors=[30, 30, 30],
            batch_size=1024,
            shuffle=False,
        )
        logging.info(
            f'{dataset_name}: loader  created for {len(phishing_indices)} nodes.'
        )

        num_nodes = data.num_nodes
        all_preds = torch.zeros(num_nodes, 1)

        with torch.no_grad():
            for batch in tqdm(phishing_loader, desc=f'{dataset_name} batch'):
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index)
                seed_nodes = batch.n_id[: batch.batch_size]
                all_preds[seed_nodes] = preds[: batch.batch_size].cpu()

        preds = all_preds[phishing_indices]
        for threshold in [0.1, 0.3, 0.5]:
            upper = dataset_name == 'IP2Location'
            accuracy = get_accuracy(preds, threshold=threshold, upper=upper)
            logging.info(
                f'{dataset_name}--Accuracy (%): {accuracy} under threshold: {threshold}'
            )


def get_accuracy(
    predictions: torch.Tensor, threshold: float, upper: bool = False
) -> float:
    total_count = 0
    positive = 0
    for pred in predictions:
        if not upper and pred <= threshold:
            positive += 1
        elif upper and pred >= threshold:
            positive += 1
        total_count += 1
    return positive / total_count


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
    )  # Map to .to_cpu()
    logging.info('In-Memory Dataset loaded.')
    weight_directory = root / cast(str, meta_args.weights_directory)

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_weak_supervision_forward(
            experiment_arg.model_args,
            dataset,
            weight_directory,
        )


if __name__ == '__main__':
    main()
