import argparse
import faulthandler
import logging
import time
from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch_geometric.data import FeatureStore, GraphStore
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore
from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore
from tgrag.gnn.model import Model
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


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
) -> None:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        loss = F.l1_loss(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1


def run_scalable_gnn(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    train_mask: Optional[torch.Tensor],
    test_mask: Optional[torch.Tensor],
    feature_store: FeatureStore,
    graph_store: GraphStore,
) -> None:
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    loader = NeighborLoader(
        data=(feature_store, graph_store),
        input_nodes=('domain', torch.arange(1000)),
        num_neighbors={('domain', 'LINKS_TO', 'domain'): [10]},
        batch_size=10,
        shuffle=True,
        num_workers=4,
    )
    logging.info('Train loader created')

    next_data = next(iter(loader))
    logging.info(f'{next_data}')

    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=128,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            train_(model=model, train_loader=loader, optimizer=optimizer)


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

    start = time.perf_counter()
    feature_store = SQLiteFeatureStore(db_path=db_path / 'graph.db')
    elapsed_1 = time.perf_counter() - start
    start = time.perf_counter()
    logging.info(f'Elapsed time accessing feature store: {elapsed_1}')
    graph_store = SQLiteGraphStore(db_path=db_path / 'graph.db')
    # elapsed_2 = time.perf_counter() - start
    # logging.info(f'Elapsed time graph store: {elapsed_2}')
    # logging.info(f'Feature store attributes: {feature_store.get_all_tensor_attrs()}')
    #
    # logging.info(f'Get the first tensor: {feature_store["domain", "x", [0]]}')
    #
    # start = time.perf_counter()
    # # TODO: Test the speed of this get_tensor and compare with other implementations
    # logging.info(
    #     f'Getting coo format of graph store: {graph_store[("domain", "LINKS_TO", "domain"), "coo"]}'
    # )
    # elapsed_3 = time.perf_counter() - start
    # logging.info(f'Elapsed time getting COO: {elapsed_3}')

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
