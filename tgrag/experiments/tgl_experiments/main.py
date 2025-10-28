import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, cast

import kuzu

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


def initialize_graph_db(
    db_path: Path, nodes_csv: Path, edges_csv: Path, buffer: int = 40
) -> Tuple[kuzu.Database, kuzu.Connection]:
    logging.info('Connecting graph storage backend')
    db = kuzu.Database(db_path, buffer_pool_size=buffer * 1024**3)
    conn = kuzu.Connection(db, num_threads=cpu_count())

    conn.execute(
        'CREATE NODE TABLE node(domain STRING, ts INT64, PRIMARY KEY(domain));'
    )
    conn.execute('CREATE REL TABLE edge(FROM node TO node, ts INT64, MANY_MANY);')
    conn.execute(f'COPY node FROM "{nodes_csv}" (HEADER=true);')
    conn.execute(f'COPY edge FROM "{edges_csv}" (HEADER=true);')
    logging.info('Graph database initialized')
    return db, conn


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    db_path = scratch / cast(str, meta_args.database_folder) / 'graphdb'
    node_path = scratch / cast(str, meta_args.node_file)
    edge_path = scratch / cast(str, meta_args.edge_file)

    db, conn = initialize_graph_db(
        db_path=db_path, nodes_csv=node_path, edges_csv=edge_path
    )

    node_df = conn.execute(
        """
        MATCH (n:node)
        RETURN n
        LIMIT 5
    """
    ).get_as_df()

    print('=== Example node records ===')
    print(node_df.head(), '\n')

    edge_df = conn.execute(
        """
        MATCH (a:node)-[r:edge]->(b:node)
        RETURN a.domain AS src, b.domain AS dst, r.ts AS ts
        LIMIT 5
    """
    ).get_as_df()

    print('=== Example edge records ===')
    print(edge_df.head())
    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')


if __name__ == '__main__':
    main()
