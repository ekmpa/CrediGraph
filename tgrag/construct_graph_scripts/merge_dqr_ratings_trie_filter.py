import csv
import logging
import multiprocessing as mp
import os
import random
import time
from typing import Any, Dict, Generator, Iterable, List, Tuple, TypedDict, cast

from tqdm import tqdm

from tgrag.utils.matching import reverse_domain
from tgrag.utils.path import get_root_dir
from tgrag.utils.prob import get_importance, get_importance_probability


class InputRow(TypedDict):
    domain: str


class OutputRow(InputRow):
    cr_score: float


class ReversedDomainTrie:
    def __init__(self) -> None:
        self.root: Dict[str, Any] = {}

    def insert(self, reversed_domain: str, score: float) -> None:
        node = self.root
        for char in reversed_domain:
            node = node.setdefault(char, {})
        node['$'] = score

    def match(self, reversed_domain: str) -> float:
        node = self.root
        best_score = -1.0
        depth = 0
        for _, char in enumerate(reversed_domain):
            if '$' in node:
                if (
                    reversed_domain == reversed_domain[:depth]
                    or reversed_domain[depth] == '.'
                ):
                    best_score = node['$']
            if char not in node:
                break
            node = node[char]
            depth += 1
        if '$' in node:
            if depth == len(reversed_domain) or reversed_domain[depth] == '.':
                best_score = node['$']
        return best_score


_trie: ReversedDomainTrie


def init_worker(trie_obj: ReversedDomainTrie) -> None:
    global _trie
    _trie = trie_obj


def estimate_csv_row_count(file_path: str) -> int:
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # exclude header


def load_dqr_cache_and_build_trie(dqr_path: str) -> ReversedDomainTrie:
    trie = ReversedDomainTrie()
    count = 0
    with open(dqr_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rev_domain = reverse_domain(row['domain'])
            trie.insert(rev_domain, float(row['pc1']))
            count += 1
    logging.info(f'Loaded {count} reversed DQR domains into Trie')
    return trie


def chunked_reader(
    reader: Iterable[InputRow], chunk_size: int
) -> Generator[List[InputRow], None, None]:
    chunk = []
    for row in reader:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def process_chunk(
    chunk: List[InputRow],
) -> Tuple[List[OutputRow], int, int]:
    pid = os.getpid()
    logging.info(f'Worker PID: {pid} processing {len(chunk)} rows ...')

    output: List[OutputRow] = []
    matched, unmatched = 0, 0

    for row in chunk:
        dom = str(row['domain']).strip()
        score = _trie.match(dom)
        row_out = row.copy()
        row_out = cast(OutputRow, row_out)
        row_out['cr_score'] = score

        if score == -1.0:
            unmatched += 1
        else:
            matched += 1

        output.append(row_out)

    logging.info(
        f'Worker PID: {pid} finished chunk. Matched: {matched}, Unmatched: {unmatched}'
    )
    return output, matched, unmatched


def merge_dqr_to_node_parallel(
    node_path: str,
    dqr_path: str,
    edges_path: str,
    output_path: str,
    filtered_edges_output_path: str,
    workers: int = 16,
    chunk_size: int = 100_000,
) -> None:
    trie = load_dqr_cache_and_build_trie(dqr_path)
    total_matched = 0
    total_unmatched = 0
    matched_domain_ids = set()

    with open(node_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('Input CSV file is missing a header row.')
        elif 'cr_score' not in reader.fieldnames:
            fieldnames = list(reader.fieldnames) + ['cr_score']
        else:
            fieldnames = list(reader.fieldnames)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

        pool = mp.Pool(workers, initializer=init_worker, initargs=(trie,))
        reader_typed = cast(Iterable[InputRow], reader)
        for result_rows, matched, unmatched in tqdm(
            pool.imap_unordered(
                process_chunk, chunked_reader(reader_typed, chunk_size)
            ),
            desc='Processing',
        ):
            total_matched += matched
            total_unmatched += unmatched
            with open(output_path, 'a', newline='', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writerows(result_rows)

            for row in result_rows:
                if row['cr_score'] != -1.0:
                    matched_domain_ids.add(
                        row['node_id']
                    )  # assumes column is named domain_id

        pool.close()
        pool.join()

    logging.info(f'Matched domains: {total_matched:,}')
    logging.info(f'Unmatched domains: {total_unmatched:,}')

    logging.info('Flushing and syncing node output file...')
    with open(output_path, 'a', encoding='utf-8') as f:
        f.flush()
        os.fsync(f.fileno())

    time.sleep(0.5)

    logging.info(f'Filtering edges to include only 1-hop neighbors...')
    filter_edges(edges_path, matched_domain_ids, filtered_edges_output_path)
    logging.info('Filtering nodes by included edges and matches...')
    filtered_nodes_output_path = output_path.replace('.csv', '_filtered.csv')
    filter_nodes_by_edges_and_matches(
        output_path,
        filtered_edges_output_path,
        filtered_nodes_output_path,
        matched_domain_ids,
    )


def get_average_importance(edges_path: str) -> float:
    with (
        open(edges_path, 'r', encoding='utf-8') as in_f,
    ):
        reader = csv.DictReader(in_f)
        importance_run: float = 0.0
        count = 0
        for row in tqdm(reader, desc='Calculating Average Importance'):
            importance_run += get_importance(
                row['pr_src'], row['hc_src'], row['pr_dst'], row['hc_dst']
            )
            count += 1
        return float(importance_run / count)


def filter_edges(
    edges_path: str,
    matched_ids: set,
    output_path: str,
) -> None:
    logging.info('Getting average importance.')
    average_importance = get_average_importance(edges_path)
    with (
        open(edges_path, 'r', encoding='utf-8') as in_f,
        open(output_path, 'w', newline='', encoding='utf-8') as out_f,
    ):
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError('Nothing to read from CSV file')
        fieldnames = list(reader.fieldnames)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        kept = 0
        for row in tqdm(reader, desc='Filtering edges'):
            if row['src'] in matched_ids or row['dst'] in matched_ids:
                writer.writerow(row)
                kept += 1
            elif random.uniform(0, 1) < get_importance_probability(
                row['pr_src'],
                row['hc_src'],
                row['pr_dst'],
                row['hc_dst'],
                average_importance,
            ):
                writer.writerow(row)
                kept += 1
        logging.info(f'Filtered edge file written with {kept:,} 1-hop edges')


def filter_nodes_by_edges_and_matches(
    scored_node_path: str,
    filtered_edge_path: str,
    output_path: str,
    matched_ids: set,
) -> None:
    edge_node_ids = set()
    with open(filtered_edge_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_node_ids.add(row['src'])
            edge_node_ids.add(row['dst'])

    keep_ids = matched_ids.union(edge_node_ids)

    with (
        open(scored_node_path, 'r', encoding='utf-8') as in_f,
        open(output_path, 'w', newline='', encoding='utf-8') as out_f,
    ):
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError('Nothing to read from CSV file')
        fieldnames = list(reader.fieldnames)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        kept = 0
        for row in tqdm(reader, desc='Filtering Nodes'):
            if row['domain'] in keep_ids:
                writer.writerow(row)
                kept += 1
        logging.info(f'Filtered node file written with {kept:,} nodes')


if __name__ == '__main__':
    root = get_root_dir()
    node_path = f'{root}/data/crawl-data/temporal/temporal_nodes.csv'
    edges_path = f'{root}/data/crawl-data/temporal/temporal_edges.csv'
    dqr_path = f'{root}/data/dqr/domain_pc1.csv'
    filtered_node_output_path = (
        f'{root}/data/crawl-data/temporal/filtered/temporal_nodes_scored.csv'
    )
    filter_edges_output_path = (
        f'{root}/data/crawl-data/temporal/filtered/temporal_edges_filtered.csv'
    )
    workers = 16
    merge_dqr_to_node_parallel(
        node_path,
        dqr_path,
        edges_path,
        filtered_node_output_path,
        filter_edges_output_path,
        workers=workers,
        chunk_size=100_000,
    )
