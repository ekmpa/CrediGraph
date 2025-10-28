import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import cast

import kuzu
import numpy as np
import pandas as pd

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


def construct_id(node_path: Path, edge_path: Path) -> None:
    verts = pd.read_csv(node_path)
    verts['id'] = np.arange(len(verts))
    id_map = dict(zip(verts['domain'], verts['id']))

    edges = pd.read_csv(edge_path)
    edges['src_id'] = edges['src'].map(id_map)
    edges['dst_id'] = edges['dst'].map(id_map)
    edges[['src_id', 'dst_id']].to_csv(
        edge_path / 'edge_with_id.csv', index=False, header=['src', 'dst']
    )
    verts[['id']].to_csv(node_path / 'nodes_with_id.csv', index=False)


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
    db = kuzu.Database('graphdb')
    conn = kuzu.Connection(db, num_threads=cpu_count())

    conn.execute('CREATE NODE TABLE node(id INT64, PRIMARY KEY(id));')
    conn.execute('CREATE REL TABLE edge(FROM node TO node, MANY_MANY);')
    conn.execute('COPY node FROM "nodes.csv" (id);')
    conn.execute('COPY edge FROM "edge_index.csv" (src, dst);')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')


if __name__ == '__main__':
    main()
