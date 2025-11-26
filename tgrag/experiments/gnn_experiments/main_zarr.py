import argparse
import logging
from typing import cast

import zarr

from tgrag.dataset.zarr_rni_dataset import ZarrDataset
from tgrag.experiments.gnn_experiments.gnn_experiment_zarr_extension import (
    run_gnn_baseline_zarr_backend,
)
from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir, get_scratch
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
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    dataset = ZarrDataset(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        seed=meta_args.global_seed,
        processed_dir=f'{scratch}/{meta_args.processed_location}',
        database_folder=f'{scratch}/{meta_args.database_folder}',
    )
    logging.info('In-Memory Zarr Dataset loaded.')
    zarr_path = scratch / cast(str, meta_args.database_folder) / 'embeddings.zarr'
    logging.info(f'Reading Zarr storage from: {zarr_path}')
    embeddings = zarr.open_array(str(zarr_path))

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_gnn_baseline_zarr_backend(
            data_arguments=experiment_arg.data_args,
            model_arguments=experiment_arg.model_args,
            weight_directory=root
            / cast(str, meta_args.weights_directory)
            / f'{meta_args.target_col}',
            dataset=dataset,
            embeddings=embeddings,
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
