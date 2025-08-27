import argparse
import logging
from typing import Dict, cast

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.categorical_encoder import CategoricalEncoder
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.norm_encoding import NormEncoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.encoders.text_encoder import TextEncoder
from tgrag.experiments.gnn_experiments.gnn_experiment import run_gnn_baseline
from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import (
    load_all_loss_tuples,
    plot_metric_across_models,
    plot_metric_per_encoder,
    plot_model_per_encoder,
)
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GNN Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn_rni/base.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
        'NORM': NormEncoder(),
        'TEXT': TextEncoder(),
        'CAT': CategoricalEncoder(),
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
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_gnn_baseline(experiment_arg.data_args, experiment_arg.model_args, dataset)
    results = load_all_loss_tuples()
    logging.info('Constructing Plots, across models')
    plot_metric_across_models(results)
    logging.info('Constructing Plots, metric per-encoder')
    plot_metric_per_encoder(results)
    logging.info('Constructing Plots, model per-encoder')
    plot_model_per_encoder(results)


if __name__ == '__main__':
    main()
