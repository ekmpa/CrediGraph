import argparse
import faulthandler
import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import FeatureStore, GraphStore
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

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


def construct_formatted_data(
    db_path: Path,
    node_csv: Path,
    dqr_csv: Path,
    seed: int = 42,
    D: int = 128,
    chunk_size: int = 1_000_000,
) -> None:
    dqr = pd.read_csv(dqr_csv)
    rng = np.random.default_rng(seed=seed)
    output_path = db_path / 'features.json'

    if output_path.exists():
        logging.info(f'{output_path} already exists, returning.')
        return

    logging.info(f'Processing {node_csv} in chunks of {chunk_size:,} rows...')
    with open(output_path, 'w') as f_out:
        for chunk in tqdm(
            pd.read_csv(node_csv, chunksize=chunk_size),
            desc='Reading vertices',
            unit='chunk',
        ):
            chunk = chunk.merge(dqr, on='domain', how='left')
            chunk['pc1'].fillna(-1.0, inplace=True)

            x_chunk = rng.normal(size=(len(chunk), D)).astype(np.float32)

            for i, (_, row) in tqdm(enumerate(chunk.iterrows()), desc='Reading Chunk'):
                record = {
                    'domain': row['domain'],
                    'ts': int(row['ts']),
                    'y': float(row['pc1']),
                    'x': x_chunk[i].tolist(),
                }
                f_out.write(json.dumps(record) + '\n')

    logging.info(f'Streaming write complete to {output_path}')


def build_domain_id_mapping(
    node_csv: Path, edge_csv: Path, out_dir: Path, chunk_size: int = 1_000_000
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nid_map_path = out_dir / 'nid_map.pkl'
    nid_array_path = out_dir / 'nid.npy'
    edges_out_path = out_dir / 'edges_with_id.csv'

    if nid_array_path.exists() and nid_map_path.exists() and edges_out_path.exists():
        logging.info(
            f'nid.npy, nid_map.pkl and edges with IDs already exists at {out_dir}, returning.'
        )
        return

    logging.info(f'Building domain to id mapping from {node_csv}...')
    domain_to_id = {}
    next_id = 0
    domain_list = []

    if not (nid_array_path.exists() and nid_map_path.exists()):
        for chunk in tqdm(
            pd.read_csv(node_csv, chunksize=chunk_size),
            desc='Reading vertices',
            unit='chunk',
        ):
            for domain in chunk['domain'].astype(str):
                if domain not in domain_to_id:
                    domain_to_id[domain] = next_id
                    domain_list.append(domain)
                    next_id += 1

        logging.info(f'Total unique domains: {len(domain_to_id):,}')
        np.save(nid_array_path, np.arange(len(domain_list), dtype=np.int64))
        with open(nid_map_path, 'wb') as f:
            pickle.dump(domain_to_id, f)

    if not edges_out_path.exists():
        logging.info(f'Rewriting {edge_csv} to {edges_out_path} with ID mapping...')
        with open(edges_out_path, 'w') as fout:
            fout.write('src_id,dst_id,ts\n')

            for chunk in tqdm(
                pd.read_csv(edge_csv, chunksize=chunk_size),
                desc='Rewriting edges',
                unit='chunk',
            ):
                chunk['src_id'] = chunk['src'].map(domain_to_id)
                chunk['dst_id'] = chunk['dst'].map(domain_to_id)

                chunk[['src_id', 'dst_id', 'ts']].astype(
                    {'src_id': 'int64', 'dst_id': 'int64'}
                ).to_csv(fout, header=False, index=False)

    logging.info(
        f'Finished. Saved nid_map.pkl, nid.npy, and edges_with_id.csv to {out_dir}'
    )


def initialize_graph_db(db_path: Path) -> sqlite3.Connection:
    logging.info('Connecting graph storage backend')
    graph_db_path = db_path / 'graph.db'
    db_path / 'edges_with_id.csv'

    if graph_db_path.exists():
        logging.info(f'Existing database found at {db_path}, skipping initalization.')
        con = sqlite3.connect(f'{graph_db_path}')
        return con

    con = sqlite3.connect(f'{graph_db_path}')
    cur = con.cursor()
    cur.execute(
        'CREATE TABLE domain(name TEXT PRIMARY KEY, ts INTEGER, x BLOB , y REAL)'
    )
    con.commit()
    logging.info('Graph database initialized')
    return con


def populate_from_json(con: sqlite3.Connection, json_path: Path) -> None:
    with open(json_path, 'r') as f:
        rows = []
        for line in tqdm(f, desc='Populating relational database with JSON'):
            if not line.strip():
                continue
            record = json.loads(line)

            x = np.array(record['x'], dtype=np.float32).tobytes()
            rows.append(
                (str(record['domain']), int(record['ts']), x, float(record['y']))
            )
            logging.info(f'Type of x: {type(x)}')
            con.execute(
                'INSERT INTO domain VALUES (?, ?, ?, ?)',
                (str(record['domain']), int(record['ts']), x, float(record['y'])),
            )
    logging.info('Database populated')
    con.commit()


def run_scalable_gnn(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    feature_store: FeatureStore,
    graph_store: GraphStore,
) -> None:
    loader = NeighborLoader(
        data=(feature_store, graph_store),
        input_nodes=('domain', train_mask),
        num_neighbors={('domain', 'link', 'domain'): model_arguments.num_neighbors},
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
    node_path = scratch / cast(str, meta_args.node_file)
    scratch / cast(str, meta_args.edge_file)
    dqr_path = root / 'data' / 'dqr' / 'domain_pc1.csv'

    # build_domain_id_mapping(node_csv=node_path, edge_csv=edge_path, out_dir=db_path)
    construct_formatted_data(db_path=db_path, node_csv=node_path, dqr_csv=dqr_path)
    con = initialize_graph_db(db_path=db_path)
    populate_from_json(con=con, json_path=db_path / 'features.json')

    logging.info('View of feature and graph store:')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')

    logging.info('Completed.')


if __name__ == '__main__':
    main()
