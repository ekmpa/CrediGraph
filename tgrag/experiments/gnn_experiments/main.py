import argparse

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
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        print(f'Running: {experiment}')
        run_gnn_baseline(experiment_arg.data_args, experiment_arg.model_args)
    results = load_all_loss_tuples()
    plot_metric_across_models(results)
    plot_metric_per_encoder(results)
    plot_model_per_encoder(results)


if __name__ == '__main__':
    main()
