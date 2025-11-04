import argparse
import faulthandler
import logging
import pickle
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, cast

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


def construct_kuzu_format(
    db_path: Path,
    node_csv: Path,
    dqr_csv: Path,
    seed: int = 42,
    D: int = 128,
    chunk_size: int = 1_000_000,
) -> None:
    dqr = pd.read_csv(dqr_csv)
    rng = np.random.default_rng(seed=seed)

    x_list = []
    y_list = []
    ts_list = []

    logging.info(f'Processing {node_csv} in chunks of {chunk_size:,} rows...')
    x_path = db_path / 'x.npy'
    y_path = db_path / 'y.npy'
    ts_path = db_path / 'ts.npy'

    if x_path.exists() and y_path.exists() and ts_path.exists():
        assert (
            np.load(x_path).dtype == np.float32
        ), f'x.npy has wrong dtype: {np.load(x_path).dtype}'
        assert (
            np.load(y_path).dtype == np.float32
        ), f'y.npy has wrong dtype: {np.load(y_path).dtype}'
        assert (
            np.load(ts_path).dtype == np.int64
        ), f'ts.npy has wrong dtype: {np.load(ts_path).dtype}'
        logging.info(f'x.npy, y.npy and ts.npy at {db_path} already exists, returning.')
        return

    for chunk in tqdm(
        pd.read_csv(node_csv, chunksize=chunk_size),
        desc='Reading vertices',
        unit='chunk',
    ):
        chunk = chunk.merge(dqr, on='domain', how='left')
        chunk['pc1'].fillna(-1.0, inplace=True)

        x_chunk = rng.normal(size=(len(chunk), D)).astype(np.float32)

        x_list.append(x_chunk)
        y_list.append(chunk['pc1'].astype(np.float32).values)
        ts_list.append(chunk['ts'].astype(np.int64).values)

    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    ts = np.concatenate(ts_list)

    logging.info(f'Saving arrays to {db_path}...')
    np.save(x_path, x)
    np.save(y_path, y)
    np.save(ts_path, ts)

    logging.info(f'Saved x{list(x.shape)}, y[{y.shape[0]}]')


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


def initialize_graph_db(
    db_path: Path, buffer: int = 40
) -> Tuple[kuzu.Database, kuzu.Connection]:
    logging.info('Connecting graph storage backend')
    graph_db_path = db_path / 'graphdb'
    edges_csv = db_path / 'edges_with_id.csv'

    if graph_db_path.exists():
        logging.info(f'Existing database found at {db_path}, skipping initalization.')
        db = kuzu.Database(graph_db_path, buffer_pool_size=buffer * 1024**3)
        conn = kuzu.Connection(db, num_threads=cpu_count())
        return db, conn

    db = kuzu.Database(graph_db_path, buffer_pool_size=buffer * 1024**3)
    conn = kuzu.Connection(db, num_threads=cpu_count())

    conn.execute(
        'CREATE NODE TABLE domain(nid INT64, x FLOAT[128], ts INT64, y FLOAT, PRIMARY KEY(nid));'
    )
    conn.execute('CREATE REL TABLE link(FROM domain TO domain, ts INT64, MANY_MANY);')
    conn.execute(
        f'COPY domain FROM ("{db_path / "nid.npy"}", "{db_path / "x.npy"}", "{db_path / "ts.npy"}", "{db_path / "y.npy"}") BY COLUMN;'
    )
    conn.execute(f'COPY link FROM "{edges_csv}" (HEADER=true);')
    logging.info('Graph database initialized')
    return db, conn


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
    edge_path = scratch / cast(str, meta_args.edge_file)
    dqr_path = root / 'data' / 'dqr' / 'domain_pc1.csv'

    build_domain_id_mapping(node_csv=node_path, edge_csv=edge_path, out_dir=db_path)
    construct_kuzu_format(db_path=db_path, node_csv=node_path, dqr_csv=dqr_path)

    db, conn = initialize_graph_db(db_path=db_path)

    try:
        df = conn.execute('MATCH (n:domain) RETURN n LIMIT 5').get_as_df()
        print(df.head())
    except RuntimeError as e:
        print('No domain table found:', e)

    node_df = conn.execute(
        """
        MATCH (n:domain)
        RETURN n.nid AS domain, n.ts AS ts, n.x as RNI
        LIMIT 5
    """
    ).get_as_df()

    print('=== Example node records ===')
    print(node_df.head(), '\n')

    edge_df = conn.execute(
        """
        MATCH (a:domain)-[r:link]->(b:domain)
        RETURN a.nid AS src, b.nid AS dst, r.ts AS ts
        LIMIT 5
    """
    ).get_as_df()

    print('=== Example edge records ===')
    print(edge_df.head())
    feature_store, graph_store = db.get_torch_geometric_remote_backend()

    logging.info(f'Feature store keys: {feature_store.get_all_tensor_attrs()}')
    logging.info(f'Graph store keys: {graph_store.get_all_edge_attrs()}')

    logging.info('View of feature and graph store:')
    try:
        y_subset = feature_store['domain', 'y', 0]
        logging.info(type(y_subset))
    except Exception as e:
        logging.exception(f'Error accessing feature store {e}')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')

    logging.info('Completed.')


if __name__ == '__main__':
    main()
