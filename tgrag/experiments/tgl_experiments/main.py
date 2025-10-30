import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, cast

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


def build_feature_and_label_arrays(
    db_path: Path, node_csv: Path, dqr_csv: Path, seed: int = 42, D: int = 128
) -> None:
    logging.info('Loading vertex and DQR data...')
    nodes = pd.read_csv(node_csv)
    dqr = pd.read_csv(dqr_csv)

    nodes = nodes.merge(dqr, on='domain', how='left')
    nodes['pc1'].fillna(-1.0, inplace=True)

    logging.info('Generating random features...')
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(size=(len(nodes), D)).astype(np.float32)

    np.save(db_path / 'domains.npy', nodes['domain'].values)
    np.save(db_path / 'x.npy', x)
    np.save(db_path / 'y.npy', nodes['pc1'].astype(np.float32).values)
    np.save(db_path / 'ts.npy', nodes['ts'].astype(np.int64).values)

    logging.info(f'Saved: x[{x.shape}], y[{nodes.shape[0]}]')


def initialize_graph_db(
    db_path: Path, nodes_csv: Path, edges_csv: Path, buffer: int = 40
) -> Tuple[kuzu.Database, kuzu.Connection]:
    logging.info('Connecting graph storage backend')

    if db_path.exists():
        logging.info(f'Existing database found at {db_path}, skipping initalization.')
        db = kuzu.Database(db_path, buffer_pool_size=buffer * 1024**3)
        conn = kuzu.Connection(db, num_threads=cpu_count())
        return db, conn

    db = kuzu.Database(db_path, buffer_pool_size=buffer * 1024**3)
    conn = kuzu.Connection(db, num_threads=cpu_count())

    conn.execute(
        f'CREATE NODE TABLE domain(name STRING, x FLOAT[128], ts INT64, y FLOAT, PRIMARY KEY(name));'
    )
    conn.execute('CREATE REL TABLE link(FROM domain TO domain, ts INT64, MANY_MANY);')
    conn.execute(
        f'COPY domain FROM "({db_path}/domains.npy, {db_path}/x.npy, {db_path}/ts.npy, {db_path}/y.npy)" BY COLUMN;'
    )
    conn.execute(f'COPY link FROM "{edges_csv}" (HEADER=true);')
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

    db_path = scratch / cast(str, meta_args.database_folder)
    node_path = scratch / cast(str, meta_args.node_file)
    edge_path = scratch / cast(str, meta_args.edge_file)

    build_feature_and_label_arrays(
        db_path=db_path,
        node_csv=node_path,
        dqr_csv=root / 'data' / 'dqr' / 'domain_pc1.csv',
    )

    db, conn = initialize_graph_db(
        db_path=db_path / 'graphdb', nodes_csv=node_path, edges_csv=edge_path
    )

    node_df = conn.execute(
        """
        MATCH (n:node)
        RETURN n.domain AS domain, n.ts AS ts
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
    feature_store, graph_store = db.get_torch_geometric_remote_backend()

    logging.info(f'Feature store keys: {feature_store.get_all_tensor_attrs()}')
    logging.info(f'Graph store keys: {graph_store.get_all_edge_attrs()}')
    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')


if __name__ == '__main__':
    main()
