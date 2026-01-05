import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, cast

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.encoders.encoder import Encoder
from tgrag.encoders.rni_encoding import RNIEncoder
from tgrag.gnn.model import Model
from tgrag.utils.args import ModelArguments, parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Return all embeddings per snapshot.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def get_embeddings(
    model_arguments: ModelArguments,
    dataset: TemporalDataset,
    weight_directory: Path,
) -> None:
    get_root_dir()
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    domain_to_idx_mapping = dataset.get_mapping()
    idx_to_domain_mapping = {
        v: k for k, v in domain_to_idx_mapping.items()
    }  # One time, to speed up lookup
    tensor_idx = torch.tensor(list(domain_to_idx_mapping.values()))
    logging.info('Mapping returned.')
    model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info('Model Loaded.')
    model.eval()

    tensor_dataset = TensorDataset(tensor_idx)

    idx_loader = DataLoader(
        dataset=tensor_dataset, batch_size=124, shuffle=True, num_workers=4
    )

    num_nodes = data.num_nodes
    all_preds_embeddings = torch.zeros(num_nodes, 256)

    all_domain_to_embeddings = {}

    for idx_batch in tqdm(idx_loader, desc='Batching Domain Index'):
        idx_batch = torch.Tensor(idx_batch)
        logging.info(f'idx_batch size: {idx_batch.size()}')
        loader = NeighborLoader(
            data,
            input_nodes=idx_batch,
            num_neighbors=[30, 30, 30],
            batch_size=1024,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'batch'):
                batch = batch.to(device)
                preds = model.get_embeddings(batch.x, batch.edge_index)
                seed_nodes = batch.n_id[: batch.batch_size]
                all_preds_embeddings[seed_nodes] = preds[: batch.batch_size].cpu()

        domain_names = [idx_to_domain_mapping[idx.item()] for idx in idx_batch]
        embeddings = all_preds_embeddings[idx_batch].tolist()
        assert len(domain_names) == len(embeddings)
        batch_dict = dict(zip(domain_names, embeddings))

        all_domain_to_embeddings.update(batch_dict)

    save_path = weight_directory / 'rni_embeddings.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(all_domain_to_embeddings, f)

    logging.info(f'Saved domain embeddings to {save_path}')


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
    }

    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoder_class = encoder_classes[value]
        encoding_dict[index] = encoder_class

    dataset = TemporalDataset(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
        processed_dir=f'{scratch}/{meta_args.processed_location}',
    )  # Map to .to_cpu()
    logging.info('In-Memory Dataset loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        get_embeddings(
            experiment_arg.model_args,
            dataset,
            weight_directory,
        )


if __name__ == '__main__':
    main()
