import pytest
import yaml

from tgrag.utils.args import ExperimentArguments, MetaArguments, parse_args
from tgrag.utils.path import get_root_dir


@pytest.fixture
def experiment_yaml(tmp_path):
    root = get_root_dir()
    config = {
        'MetaArguments': {
            'log_file_path': 'logs/exp.log',
            'node_file': 'temporal_nodes.csv',
            'edge_file': 'temporal_edges.csv',
            'global_seed': 42,
        },
        'ExperimentArguments': {
            'exp_args': {
                'exp1': {
                    'model_args': {
                        'model': 'GCN',
                        'encoder': 'RNI',
                        'encoder_col': 'random',
                        'num_layers': 3,
                        'hidden_channels': 128,
                        'dropout': 0.5,
                        'lr': 0.01,
                        'epochs': 100,
                        'runs': 5,
                        'device': 0,
                        'log_steps': 10,
                    },
                    'data_args': {
                        'task_name': 'node-regression',
                        'is_regression': True,
                        'num_test_shards': 3,
                    },
                }
            }
        },
    }
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.dump(config))
    return path


def test_parse_experiment_args(experiment_yaml):
    meta_args, exp_args = parse_args(experiment_yaml)

    assert isinstance(meta_args, MetaArguments)
    assert isinstance(exp_args, ExperimentArguments)

    assert meta_args.global_seed == 42
    assert 'exp1' in exp_args.exp_args

    exp1 = exp_args.exp_args['exp1']

    assert exp1.model_args.model == 'GCN'
    assert exp1.data_args.task_name == 'node-regression'
    assert 'temporal_nodes.csv' in meta_args.node_file
    assert 'temporal_edges.csv' in meta_args.edge_file
