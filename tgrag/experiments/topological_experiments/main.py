import argparse

from tgrag.experiments.topological_experiments.topological_experiment import (
    run_topological_experiment,
)
from tgrag.utils.args import parse_args
from tgrag.utils.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Topological Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/topology/topological.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        print(f'Running: {experiment}')
        results = run_topological_experiment(experiment_arg.data_args, experiment)
        print(results)


if __name__ == '__main__':
    main()
