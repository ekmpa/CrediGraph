import csv
import glob
import logging
import os
import re
import sqlite3
from typing import Dict, Optional

from tqdm import tqdm

MONTH_MAP = {
    'jan': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'may': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12',
}


def extract_time_id(filename: str) -> Optional[str]:
    match = re.search(r'cc-main-(\d{4})-([a-z]{3})', filename)
    if match:
        year = match.group(1)
        month_str = match.group(2).lower()
        month_num = MONTH_MAP.get(month_str, '01')
        return f'{year}{month_num}01'
    return None


def index_rank_file_to_sqlite(rank_file: str, db_path: str) -> None:
    if os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS ranks (domain TEXT PRIMARY KEY, pr_val REAL, hc_val REAL)'
    )
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=MEMORY')

    logging.info(
        f'Indexing PageRank file: {os.path.basename(rank_file)} → {os.path.basename(db_path)}'
    )
    with open(rank_file, 'r') as f:
        buffer = []
        for line in tqdm(f, desc='Indexing PR/HC file', unit='lines'):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                domain = parts[4]
                try:
                    pr_val = float(parts[3])
                    hc_val = float(parts[1])  # Assuming hc_val is in column 2
                    buffer.append((domain, pr_val, hc_val))
                    if len(buffer) >= 100_000:
                        cur.executemany(
                            'INSERT OR IGNORE INTO ranks (domain, pr_val, hc_val) VALUES (?, ?, ?)',
                            buffer,
                        )
                        buffer = []
                except ValueError:
                    continue
        if buffer:
            cur.executemany(
                'INSERT OR IGNORE INTO ranks (domain, pr_val, hc_val) VALUES (?, ?, ?)',
                buffer,
            )

    conn.commit()
    conn.close()


def count_lines(filepath: str) -> int:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def merge_vertices_rank_centrality(folder_path: str) -> None:
    logging.info(f'Processing folder: {folder_path}')
    folder_path = os.path.abspath(folder_path)

    vertex_files = sorted(glob.glob(os.path.join(folder_path, '*domain-vertices.txt')))
    edge_files = sorted(glob.glob(os.path.join(folder_path, '*domain-edges.txt')))
    rank_files = sorted(glob.glob(os.path.join(folder_path, '*domain-ranks.txt')))

    time_id_to_db: Dict[str, str] = {}
    for rank_file in rank_files:
        time_id = extract_time_id(os.path.basename(rank_file))
        if time_id:
            db_path = os.path.join(folder_path, f'rank_index_{time_id}.sqlite')
            index_rank_file_to_sqlite(rank_file, db_path)
            time_id_to_db[time_id] = db_path

    nodes_out_path = os.path.join(folder_path, 'temporal_nodes.csv')
    edges_out_path = os.path.join(folder_path, 'temporal_edges.csv')

    with (
        open(nodes_out_path, 'w', newline='') as nodes_out,
        open(edges_out_path, 'w', newline='') as edges_out,
    ):
        node_writer = csv.writer(nodes_out)
        edge_writer = csv.writer(edges_out)

        node_writer.writerow(['domain', 'node_id', 'time_id', 'pr_val', 'hc_val'])
        edge_writer.writerow(['src', 'dst', 'time_id'])

        for vfile in vertex_files:
            time_id = extract_time_id(os.path.basename(vfile))
            if not time_id:
                continue
            db_path = time_id_to_db.get(time_id, '')
            if db_path:
                continue

            logging.info(f'Processing vertex file: {os.path.basename(vfile)}')
            total_lines = count_lines(vfile)

            conn = sqlite3.connect(db_path)
            conn.execute('PRAGMA journal_mode=OFF')
            conn.execute('PRAGMA synchronous=OFF')
            conn.execute('PRAGMA temp_store=MEMORY')
            cur = conn.cursor()

            with open(vfile, 'r') as vf:
                for line in tqdm(
                    vf, total=total_lines, desc='Writing nodes', unit='lines'
                ):
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    node_id, domain = parts[0], parts[1]
                    cur.execute(
                        'SELECT pr_val, hc_val FROM ranks WHERE domain = ?', (domain,)
                    )
                    result = cur.fetchone()
                    pr_val = result[0] if result else -1.0
                    hc_val = result[1] if result else -1.0
                    node_writer.writerow([domain, node_id, time_id, pr_val, hc_val])

            conn.close()

        for efile in edge_files:
            time_id = extract_time_id(os.path.basename(efile))
            if not time_id:
                continue

            logging.info(f'Processing edge file: {os.path.basename(efile)}')
            total_lines = count_lines(efile)

            with open(efile, 'r') as ef:
                for line in tqdm(
                    ef, total=total_lines, desc='Writing edges', unit='lines'
                ):
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    src, dst = parts[0], parts[1]
                    edge_writer.writerow([src, dst, time_id])

    logging.info('Done.')
