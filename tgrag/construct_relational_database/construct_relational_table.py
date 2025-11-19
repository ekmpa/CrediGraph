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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.rd_utils import table_has_data
from tgrag.utils.seed import seed_everything
from tgrag.utils.target_generation import strict_exact_etld1_match

parser = argparse.ArgumentParser(
    description='Construct Graph Relational Table.',
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
    dqr_df = pd.read_csv(dqr_csv)
    dqr_domains = {
        row['domain']: row  # full row; row["pc1"] is what you need
        for _, row in dqr_df.iterrows()
    }

    rng = np.random.default_rng(seed=seed)
    output_path = db_path / 'features.json'

    if output_path.exists():
        logging.info(f'{output_path} already exists, returning.')
        return

    logging.info(f'Processing {node_csv} in chunks of {chunk_size:,} rows...')

    included: int = 0
    with open(output_path, 'w') as f_out:
        for chunk in tqdm(
            pd.read_csv(node_csv, chunksize=chunk_size),
            desc='Reading vertices',
            unit='chunk',
        ):
            x_chunk = rng.normal(size=(len(chunk), D)).astype(np.float32)

            for i, (_, row) in tqdm(
                enumerate(chunk.iterrows()), desc='Iterating chunk'
            ):
                raw_domain = str(row['domain']).strip()

                etld1 = strict_exact_etld1_match(raw_domain, dqr_domains)

                if etld1 is None:
                    y = -1.0
                else:
                    included += 1
                    y = float(dqr_domains[etld1]['pc1'])

                record = {
                    'domain': raw_domain,
                    'ts': int(row['ts']),
                    'y': y,
                    'x': x_chunk[i].tolist(),
                }

                f_out.write(json.dumps(record) + '\n')

    logging.info(f'There are {included} domains that exist in DQR')
    logging.info(f'Streaming write complete to {output_path}')


def initialize_graph_db(db_path: Path) -> sqlite3.Connection:
    logging.info('Connecting graph storage backend')
    graph_db_path = db_path / 'graph.db'

    con = sqlite3.connect(f'{graph_db_path}')
    cur = con.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS domain(id INTEGER PRIMARY KEY, ts INTEGER, x BLOB , y REAL)'
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS edges (
        src_id INTEGER,
        dst_id INTEGER,
        relation TEXT,
        ts INTEGER
    )
    """
    )

    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id)')

    con.commit()
    logging.info('Graph database initialized')
    return con


def construct_masks_from_json(
    nid_map_path: Path, json_path: Path, db_path: Path, seed: int = 0
) -> None:
    output_path = db_path / 'split_idx.pt'
    if output_path.exists():
        logging.info(f'{output_path} already exists, returning.')
        return

    with open(nid_map_path, 'rb') as f:
        domain_to_id = pickle.load(f)
    labeled_idx = []
    labeled_scores = []

    with open(json_path, 'r') as f:
        for line in tqdm(f, desc='Constructing Masks'):
            if not line.strip():
                continue

            record = json.loads(line)
            domain = str(record['domain'])

            if domain not in domain_to_id:
                logging.warning(f'Domain {domain} not in mapping; skipping.')
                continue

            node_id = int(domain_to_id[domain])
            y = float(record['y'])

            # skip unlabeled
            if y < 0:
                continue

            labeled_idx.append(node_id)
            labeled_scores.append(y)

    if len(labeled_idx) == 0:
        raise RuntimeError('No labeled nodes found.')

    labeled_idx = np.array(labeled_idx)
    labeled_scores = np.array(labeled_scores)

    quantiles = np.quantile(labeled_scores, [1 / 3, 2 / 3])

    quartile_labels = np.digitize(labeled_scores, bins=quantiles)

    train_idx, temp_idx, _, quartile_labels_temp = train_test_split(
        labeled_idx,
        quartile_labels,
        train_size=0.6,
        stratify=quartile_labels,
        random_state=seed,
    )

    valid_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        stratify=quartile_labels_temp,
        random_state=seed,
    )

    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    valid_idx = torch.as_tensor(valid_idx, dtype=torch.long)
    test_idx = torch.as_tensor(test_idx, dtype=torch.long)

    logging.info(f'Train size: {train_idx.size(0)}')
    logging.info(f'Valid size: {valid_idx.size(0)}')
    logging.info(f'Test size: {test_idx.size(0)}')

    torch.save({'train': train_idx, 'valid': valid_idx, 'test': test_idx}, output_path)
    logging.info(f'Saved splits at {output_path}')


def populate_edges(
    con: sqlite3.Connection, edges_path: Path, chunk_size: int = 1_000_000
) -> None:
    if not table_has_data(con=con, table='edges'):
        logging.info(f'Populating edges from {edges_path} using pandas chunks...')
        for chunk in tqdm(
            pd.read_csv(edges_path, chunksize=chunk_size),
            desc='Populating edges',
            unit='chunk',
        ):
            chunk['relation'] = 'LINKS_TO'
            data = (
                chunk[['src_id', 'dst_id', 'relation', 'ts']]
                .astype({'src_id': 'int64', 'dst_id': 'int64', 'ts': 'int64'})
                .to_records(index=False)
                .tolist()
            )
            con.executemany('INSERT INTO edges VALUES (?, ?, ?, ?)', data)
            con.commit()
    else:
        logging.info('Edge table already populated skipping...')


def populate_from_json(
    con: sqlite3.Connection, nid_map_path: Path, json_path: Path
) -> None:
    if not table_has_data(con=con, table='domain'):
        with open(nid_map_path, 'rb') as f:
            domain_to_id = pickle.load(f)
        logging.info(f'Loaded {len(domain_to_id):,} domain-id mappings')

        with open(json_path, 'r') as f:
            for line in tqdm(f, desc='Populating relational database with JSON'):
                if not line.strip():
                    continue
                record = json.loads(line)

                domain = str(record['domain'])
                if domain not in domain_to_id:
                    logging.warning(f'Domain {domain} not found in mapping; skipping.')
                    continue

                id = int(domain_to_id[domain])

                x = np.array(record['x'], dtype=np.float32).tobytes()
                con.execute(
                    'INSERT INTO domain VALUES (?, ?, ?, ?)',
                    (id, int(record['ts']), x, float(record['y'])),
                )
        logging.info('Database populated')
        con.commit()
    else:
        logging.info('Domain table is already populated skipping...')


def main() -> None:
    faulthandler.enable()
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, _ = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    db_path = scratch / cast(str, meta_args.database_folder)
    node_path = scratch / cast(str, meta_args.node_file)
    dqr_path = root / 'data' / 'dqr' / 'domain_pc1.csv'

    construct_formatted_data(db_path=db_path, node_csv=node_path, dqr_csv=dqr_path)
    construct_masks_from_json(
        nid_map_path=db_path / 'nid_map.pkl',
        json_path=db_path / 'features.json',
        db_path=db_path,
        seed=meta_args.global_seed,
    )
    con = initialize_graph_db(db_path=db_path)
    populate_from_json(
        con=con,
        nid_map_path=db_path / 'nid_map.pkl',
        json_path=db_path / 'features.json',
    )
    populate_edges(con=con, edges_path=db_path / 'edges_with_id.csv')
    logging.info('Completed.')


if __name__ == '__main__':
    main()
