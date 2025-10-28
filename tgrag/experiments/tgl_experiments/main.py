import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import cast

import kuzu
import numpy as np
import pandas as pd
from tqdm import tqdm

from tgrag.utils.args import parse_args
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


def construct_id(node_path: Path, edge_path: Path, chunksize: int = 5_000_000) -> None:
    node_out = node_path.parent / 'nodes_with_id.csv'
    edge_out = edge_path.parent / 'edges_with_id.csv'

    if node_out.exists() and edge_out.exists():
        logging.info(f'Found existing {node_out.name} and {edge_out.name}, skipping.')
        return

    logging.info('Reading nodes and assigning IDs...')
    verts = pd.read_csv(node_path)
    verts['id'] = np.arange(len(verts))
    id_map = dict(zip(verts['domain'], verts['id']))

    logging.info(f'Writing node IDs to: {node_out}')
    verts[['id']].to_csv(node_out, index=False)

    logging.info('Mapping and writing edges with IDs...')
    with open(edge_out, 'w') as fout:
        fout.write('src,dst\n')
        total_lines = sum(1 for _ in open(edge_path)) - 1  # minus header
        reader = pd.read_csv(edge_path, chunksize=chunksize)

        for chunk in tqdm(
            reader, total=total_lines // chunksize + 1, desc='Processing edges'
        ):
            chunk['src_id'] = chunk['src'].map(id_map)
            chunk['dst_id'] = chunk['dst'].map(id_map)
            chunk[['src_id', 'dst_id']].to_csv(fout, index=False, header=False)

    logging.info(f'Completed writing: {edge_out}')


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    logging.info('Mapping domains to unique IDs')
    construct_id(
        node_path=scratch / cast(str, meta_args.node_file),
        edge_path=scratch / cast(str, meta_args.edge_file),
    )

    logging.info('Connecting graph storage backend')
    db_path = scratch / cast(str, meta_args.database_folder) / 'graphdb'
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db, num_threads=cpu_count())

    conn.execute('CREATE NODE TABLE node(id INT64, PRIMARY KEY(id));')
    conn.execute('CREATE REL TABLE edge(FROM node TO node, MANY_MANY);')
    conn.execute(f'COPY node FROM "{db_path}/nodes_with_id.csv" (id);')
    conn.execute(f'COPY edge FROM "{db_path}/edges_with_id.csv" (src, dst);')
    logging.info('Graph database initialized')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')


if __name__ == '__main__':
    main()
