import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import yaml
from hf_argparser import HfArgumentParser

from tgrag.utils.path import get_root_dir


@dataclass
class MetaArguments:
    log_file_path: Optional[str] = field(
        metadata={'help': 'Path to the log file to use.'},
    )
    global_seed: int = field(
        default=1337,
        metadata={'help': 'Random seed to use for reproducibiility.'},
    )

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass
class DataArguments:
    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    is_regression: bool = field(
        metadata={'help': 'Is the task a regression or classification problem'},
    )
    node_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    edge_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    num_test_shards: int = field(
        metadata={'help': 'Number of test splits to do for uncertainty estimates.'},
        default=1,
    )

    def __post_init__(self) -> None:
        root_dir = get_root_dir()

        def resolve_paths(files: Union[str, List[str]]) -> Union[str, List[str]]:
            if isinstance(files, str):
                return str(root_dir / files)
            return [str(root_dir / file) for file in files]

        self.node_file = resolve_paths(self.node_file)
        self.edge_file = resolve_paths(self.edge_file)


@dataclass
class ModelArguments:
    model: str = field(
        default='GCN',
        metadata={'help': 'Model identifer for the GNN.'},
    )
    encoder: str = field(
        default='RNI', metadata={'help': 'Encoder identifer for node encoding.'}
    )
    encoder_col: str = field(
        default='random',
        metadata={'help': 'The column for which the encoder will encoder.'},
    )
    num_layers: int = field(
        default=3,
        metadata={'help': 'Number of layers in GNN or iterations in message passing.'},
    )
    hidden_channels: int = field(
        default=256, metadata={'help': 'Inner dimension of update weight matrix.'}
    )
    dropout: float = field(default=0.5, metadata={'help': 'Dropout value.'})
    lr: float = field(default=0.01, metadata={'help': 'Learning Rate.'})
    epochs: int = field(default=500, metadata={'help': 'Number of epochs.'})
    runs: int = field(default=100, metadata={'help': 'Number of trials.'})
    device: int = field(default=0, metadata={'help': 'Device to be used.'})
    log_steps: int = field(
        default=50, metadata={'help': 'Step mod epoch to print logger.'}
    )


@dataclass
class ExperimentArgument:
    data_args: DataArguments = field(
        metadata={'help': 'Data arguments for GNN configuration.'}
    )
    model_args: ModelArguments = field(
        metadata={'help': 'Model arguments for the GNN.'}
    )


@dataclass
class ExperimentArguments:
    exp_args: Dict[str, ExperimentArgument] = field(
        metadata={'help': 'List of experiments.'}
    )

    def __post_init__(self) -> None:
        def _remap_experiment_args(
            experiments: Dict[str, ExperimentArgument],
        ) -> Dict[str, ExperimentArgument]:
            for exp_name, exp_val in experiments.items():
                if isinstance(exp_val, dict):
                    model_args = ModelArguments(**exp_val['model_args'])
                    data_args = DataArguments(**exp_val['data_args'])
                    experiments[exp_name] = ExperimentArgument(
                        model_args=model_args,
                        data_args=data_args,
                    )
            return experiments

        self.exp_args = _remap_experiment_args(self.exp_args)


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[MetaArguments, ExperimentArguments]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = config_dict['MetaArguments'] | config_dict['ExperimentArguments']
    parser = HfArgumentParser((MetaArguments, ExperimentArguments))
    return parser.parse_dict(config_dict, allow_extra_keys=True)
