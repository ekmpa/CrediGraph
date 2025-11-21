import argparse
import faulthandler
import logging
import time
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch_geometric.data import FeatureStore, GraphStore
from torch_geometric.loader import NeighborLoader, NodeLoader
from tqdm import tqdm

from tgrag.dataset.sampler import SQLiteNeighborSampler
from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore
from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore
from tgrag.gnn.model import Model
from tgrag.utils.args import DataArguments, ModelArguments, parse_args
from tgrag.utils.logger import Logger, setup_logging
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
    train_mask: torch.Tensor,
    optimizer: torch.optim.AdamW,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        idx = batch['domain']['id']
        preds = model(
            batch['domain']['x'], batch['domain', 'LINKS_TO', 'domain'].edge_index
        ).squeeze()
        mask = torch.isin(idx, train_mask)
        if not mask.any():
            continue
        targets = batch['domain']['y']
        loss = F.l1_loss(preds[mask], targets[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NodeLoader,
    idx_mask: torch.Tensor,
) -> Tuple[float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    total_batches = 0
    for batch in loader:
        batch = batch.to(device)
        idx = batch['domain']['id']
        preds = model(
            batch['domain']['x'], batch['domain', 'LINKS_TO', 'domain'].edge_index
        ).squeeze()
        mask = torch.isin(idx, idx_mask)
        if not mask.any():
            continue

        if (
            preds.numel() == 1
            and batch['domain', 'LINKS_TO', 'domain'].edge_index.numel() == 0
        ):
            # This is the edge case in which we get batch size == 1, and an isolated node
            continue

        targets_mask = batch['domain']['y'][mask]
        preds_mask = preds[mask]
        mean_preds = torch.full(targets_mask.size(), 0.546).to(device)
        random_preds = torch.rand(targets_mask.size()).to(device)
        loss = F.l1_loss(preds_mask, targets_mask)
        mean_loss = F.l1_loss(mean_preds, targets_mask)
        random_loss = F.l1_loss(random_preds, targets_mask)

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        total_batches += 1

    mse = total_loss / total_batches
    mse_mean = total_mean_loss / total_batches
    mse_random = total_random_loss / total_batches

    return (mse, mse_mean, mse_random)


def run_scalable_gnn(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    train_mask: Optional[torch.Tensor],
    test_mask: Optional[torch.Tensor],
    feature_store: FeatureStore,
    graph_store: GraphStore,
) -> None:
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device found: {device}')
    device = torch.device(device)
    sampler = SQLiteNeighborSampler(
        graph_store=graph_store, num_neighbors={('domain', 'LINKS_TO', 'domain'): [5]}
    )

    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=sampler,
        input_nodes=('domain', train_mask),
        batch_size=10,
    )

    test_loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=sampler,
        input_nodes=('domain', test_mask),
        batch_size=10,
    )
    logging.info('Train loader created')

    next_data = next(iter(loader))
    logging.info(f'{next_data}')

    logger = Logger(model_arguments.runs)
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
            train_(
                model=model,
                train_loader=loader,
                train_mask=train_mask,
                optimizer=optimizer,
            )
            epoch_mse_train, epoch_baseline_mse_train, epoch_random_mse_train = (
                evaluate(
                    model=model,
                    loader=loader,
                    idx_mask=train_mask,
                )
            )
            epoch_mse_test, epoch_baseline_mse_test, epoch_random_mse_test = evaluate(
                model=model,
                loader=test_loader,
                idx_mask=test_mask,
            )

            result = (
                epoch_mse_train,
                0,
                epoch_mse_test,
                epoch_baseline_mse_test,
                epoch_random_mse_test,
            )
            logger.add_result(
                run=run,
                result=(
                    epoch_mse_train,
                    0,
                    epoch_mse_test,
                    epoch_baseline_mse_test,
                ),
            )

    logging.info(logger.get_avg_statistics())
    logging.info(logger.get_statistics())


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

    logging.info(f'Loading mask from {db_path / "split_idx.pt"}')
    split_idx = torch.load(db_path / 'split_idx.pt')
    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_scalable_gnn(
            data_arguments=experiment_arg.data_args,
            model_arguments=experiment_arg.model_args,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            feature_store=feature_store,
            graph_store=graph_store,
        )

    logging.info('Completed.')


if __name__ == '__main__':
    main()
