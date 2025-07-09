import argparse

from tgrag.experiments.correlation_experiments.pr_correlation_experiment import (
    run_pr_correlation,
    run_pr_cr_bin_correlation,
)
from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Correlation Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/pr_correlation/base.yaml',
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
        if experiment_arg.model_args.model == 'bin':
            run_pr_cr_bin_correlation(experiment_arg.data_args)
        else:
            run_pr_correlation(experiment_arg.data_args)


if __name__ == '__main__':
    main()
