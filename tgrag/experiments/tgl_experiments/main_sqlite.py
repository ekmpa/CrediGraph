import argparse
import faulthandler
import logging
from typing import Optional, cast

import torch
from torch_geometric.data import FeatureStore, GraphStore
from torch_geometric.loader import NeighborLoader

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore
from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore
from tgrag.utils.args import DataArguments, ModelArguments, parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGL Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/tgl/base.yaml',
    help='Path to yaml configuration file to use',
)


def run_scalable_gnn(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    train_mask: Optional[torch.Tensor],
    test_mask: Optional[torch.Tensor],
    feature_store: FeatureStore,
    graph_store: GraphStore,
) -> None:
    loader = NeighborLoader(
        data=(feature_store, graph_store),
        input_nodes=('domain', torch.arange(100)),
        num_neighbors={('domain', 'LINKS_TO', 'domain'): model_arguments.num_neighbors},
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Train loader created')

    next_data = next(iter(loader))
    logging.info(f'{next_data}')


def main() -> None:
    faulthandler.enable()
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    db_path = scratch / cast(str, meta_args.database_folder)

    logging.info('View of feature and graph store:')

    feature_store = SQLiteFeatureStore(db_path=db_path / 'graph.db')
    graph_store = SQLiteGraphStore(db_path=db_path / 'graph.db')
    logging.info(f'Feature store attributes: {feature_store.get_all_tensor_attrs()}')

    logging.info(
        f'Get the first tensor: {feature_store["domain", "x", [0, 3, 5, 100]]}'
    )

    # TODO: Test the speed of this get_tensor and compare with other implementations
    logging.info(
        f'Getting coo format of graph store: {graph_store[("domain", "LINKS_TO", "domain"), "coo"]}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_scalable_gnn(
            data_arguments=experiment_arg.data_args,
            model_arguments=experiment_arg.model_args,
            train_mask=None,
            test_mask=None,
            feature_store=feature_store,
            graph_store=graph_store,
        )

    logging.info('Completed.')


if __name__ == '__main__':
    main()
