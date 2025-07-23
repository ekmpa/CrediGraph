import argparse
import logging
from typing import Dict, cast

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.norm_encoding import NormEncoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.experiments.gnn_experiments.gnn_experiment import run_gnn_baseline
from tgrag.experiments.gnn_experiments.gnn_experiment_full_batch import (
    run_gnn_baseline_full_batch,
)
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

ENCODER_CLASSES: Dict[str, Encoder] = {
    'RNI': RNIEncoder(),
    'NORM': NormEncoder(),
}

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
parser.add_argument(
    '--full-batch',
    action='store_true',
    help='Whether to use full-batching. Mini-batching is by default.',
)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoder_class = ENCODER_CLASSES[value]
        encoding_dict[index] = encoder_class

    dataset = TemporalDataset(
        root=f'{root}/data/crawl-data/temporal',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        encoding=encoding_dict,
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'Running: {experiment}')
        if args.full_batch:
            run_gnn_baseline_full_batch(
                experiment_arg.data_args, experiment_arg.model_args, dataset
            )
        else:
            run_gnn_baseline(
                experiment_arg.data_args, experiment_arg.model_args, dataset
            )
    results = load_all_loss_tuples()
    logging.info('Constructing Plots, across models')
    plot_metric_across_models(results)
    logging.info('Constructing Plots, metric per-encoder')
    plot_metric_per_encoder(results)
    logging.info('Constructing Plots, model per-encoder')
    plot_model_per_encoder(results)


if __name__ == '__main__':
    main()
