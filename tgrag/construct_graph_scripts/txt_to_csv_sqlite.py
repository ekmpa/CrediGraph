import csv
import glob
import logging
import os
import re
import sqlite3
from typing import Dict, Optional, cast

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


def index_rank_file_to_sqlite(rank_file: str, db_rank_path: str) -> None:
    if os.path.exists(db_rank_path):
        return

    conn = sqlite3.connect(db_rank_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS ranks ( domain TEXT PRIMARY KEY, pr_val REAL, hc_val REAL)'
    )
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=MEMORY')

    logging.info(
        f'Indexing PageRank file: {os.path.basename(rank_file)} → {os.path.basename(db_rank_path)}'
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


def index_node_file_to_sqlite(node_file: str, db_rank_path: str) -> None:
    if os.path.exists(db_rank_path):
        return

    conn = sqlite3.connect(db_rank_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS node_map (node_id TEXT PRIMARY KEY, domain TEXT)'
    )
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=MEMORY')

    logging.info(
        f'Indexing Vertex file: {os.path.basename(node_file)} → {os.path.basename(db_rank_path)}'
    )
    with open(node_file, 'r') as f:
        buffer = []
        for line in tqdm(f, desc='Indexing node file', unit='lines'):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                node_id = parts[0]
                try:
                    domain = parts[1]
                    buffer.append((node_id, domain))
                    if len(buffer) >= 100_000:
                        cur.executemany(
                            'INSERT OR IGNORE INTO node_map (node_id, domain) VALUES (?, ?)',
                            buffer,
                        )
                        buffer = []
                except ValueError:
                    continue
        if buffer:
            cur.executemany(
                'INSERT OR IGNORE INTO node_map (node_id, domain) VALUES (?, ?)',
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

    time_id_to_db_rank: Dict[str, str] = {}
    time_id_to_db_node: Dict[str, str] = {}
    logging.info('Indexing Rank files.')
    for rank_file in rank_files:
        time_id = extract_time_id(os.path.basename(rank_file))
        if time_id:
            db_rank_path = os.path.join(folder_path, f'rank_index_{time_id}.sqlite')
            index_rank_file_to_sqlite(rank_file, db_rank_path)
            time_id_to_db_rank[time_id] = db_rank_path
    logging.info('Indexing Vertex files.')
    for vertex_file in vertex_files:
        time_id = extract_time_id(os.path.basename(vertex_file))
        if time_id:
            db_node_path = os.path.join(folder_path, f'vertex_index_{time_id}.sqlite')
            index_node_file_to_sqlite(vertex_file, db_node_path)
            time_id_to_db_node[time_id] = db_node_path

    nodes_out_path = os.path.join(folder_path, 'temporal_nodes.csv')
    edges_out_path = os.path.join(folder_path, 'temporal_edges.csv')

    with (
        open(nodes_out_path, 'w', newline='') as nodes_out,
        open(edges_out_path, 'w', newline='') as edges_out,
    ):
        node_writer = csv.writer(nodes_out)
        edge_writer = csv.writer(edges_out)

        node_writer.writerow(['domain', 'node_id', 'time_id', 'pr_val', 'hc_val'])
        edge_writer.writerow(
            [
                'src',
                'dst',
                'time_id',
                'pr_val_src',
                'hc_val_src',
                'pr_val_dst',
                'hc_val_dst',
            ]
        )

        for vfile in vertex_files:
            time_id = extract_time_id(os.path.basename(vfile))
            db_rank_path = time_id_to_db_rank.get(cast(str, time_id), '')

            logging.info(f'Processing vertex file: {os.path.basename(vfile)}')
            total_lines = count_lines(vfile)

            conn = sqlite3.connect(db_rank_path)
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
            db_rank_path = time_id_to_db_rank.get(time_id, '')
            db_node_path = time_id_to_db_node.get(time_id, '')
            logging.info(f'Processing edge file: {os.path.basename(efile)}')
            total_lines = count_lines(efile)

            conn_rank = sqlite3.connect(db_rank_path)
            conn_rank.execute('PRAGMA journal_mode=OFF')
            conn_rank.execute('PRAGMA synchronous=OFF')
            conn_rank.execute('PRAGMA temp_store=MEMORY')
            cur_rank = conn_rank.cursor()

            domain_to_rank = {}
            cur_rank.execute('SELECT domain, pr_val, hc_val from ranks')
            for domain, pr, hc in cur_rank.fetchall():
                domain_to_rank[domain] = (pr, hc)

            conn_node = sqlite3.connect(db_node_path)
            conn_node.execute('PRAGMA journal_mode=OFF')
            conn_node.execute('PRAGMA synchronous=OFF')
            conn_node.execute('PRAGMA temp_store=MEMORY')
            cur_node = conn_node.cursor()

            with open(efile, 'r') as ef:
                for line in tqdm(
                    ef, total=total_lines, desc='Writing edges', unit='lines'
                ):
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    src, dst = parts[0], parts[1]
                    cur_node.execute(
                        'SELECT domain FROM node_map WHERE node_id = ?', (src,)
                    )
                    src_domain = cur_node.fetchone()
                    pr_src, hc_src = (
                        domain_to_rank.get(src_domain[0], (-1.0, -1.0))
                        if src_domain
                        else (-1.0, -1.0)
                    )

                    cur_node.execute(
                        'SELECT domain FROM node_map WHERE node_id = ?', (dst,)
                    )
                    dst_domain = cur_rank.fetchone()
                    pr_dst, hc_dst = (
                        domain_to_rank.get(dst_domain[0], (-1.0, -1.0))
                        if dst_domain
                        else (-1.0, -1.0)
                    )
                    edge_writer.writerow(
                        [src, dst, time_id, pr_src, hc_src, pr_dst, hc_dst]
                    )
            cur_rank.close()
            cur_node.close()

    logging.info('Done.')
