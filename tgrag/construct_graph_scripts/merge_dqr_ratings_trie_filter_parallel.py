import csv
import logging
import multiprocessing as mp
import os
import random
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Set,
    Tuple,
    TypedDict,
    cast,
)

from tqdm import tqdm

from tgrag.utils.matching import reverse_domain
from tgrag.utils.prob import (
    get_importance_node,
    get_importance_probability_node,
)

# ------------------------- Types -------------------------


class InputRow(TypedDict):
    domain: str
    pr_val: str
    hc_val: str


class OutputRow(InputRow):
    cr_score: float


# ------------------------- Trie -------------------------


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


# ------------------------- Globals for Workers -------------------------

_trie: ReversedDomainTrie
_matched_ids: Set[str]
_random_ids: Set[str]
_average_importance: float
_keep_ids: Set[str]


# ------------------------- Initialization -------------------------


def init_worker(trie_obj: ReversedDomainTrie) -> None:
    global _trie
    _trie = trie_obj


def init_edge_worker(matched_ids: Set[str], random_ids: Set[str]) -> None:
    global _matched_ids, _random_ids
    _matched_ids = matched_ids
    _random_ids = random_ids


def init_node_worker(keep_ids: Set[str]) -> None:
    global _keep_ids
    _keep_ids = keep_ids


# ------------------------- Utilities -------------------------


def chunked_reader(
    reader: Iterable[Dict[str, str]], chunk_size: int
) -> Generator[List[Dict[str, str]], None, None]:
    chunk = []
    for row in reader:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ------------------------- Matching -------------------------


def process_chunk(chunk: List[InputRow]) -> Tuple[List[OutputRow], int, int]:
    pid = os.getpid()
    logging.info(f'Worker PID: {pid} processing {len(chunk)} rows ...')

    output: List[OutputRow] = []
    matched, unmatched = 0, 0

    for row in chunk:
        dom = str(row['domain']).strip()
        pr_val = str(row['pr_val'])
        hc_val = str(row['hc_val'])
        score = _trie.match(dom)
        row_out = cast(OutputRow, row.copy())
        row_out['cr_score'] = score
        row_out['pr_val'] = pr_val
        row_out['hc_val'] = hc_val

        if score == -1.0:
            unmatched += 1
        else:
            matched += 1

        output.append(row_out)

    logging.info(
        f'Worker PID: {pid} finished chunk. Matched: {matched}, Unmatched: {unmatched}'
    )
    return output, matched, unmatched


# ------------------------- Importance -------------------------


def process_importance_chunk(chunk: List[Dict[str, str]]) -> Tuple[float, int]:
    pid = os.getpid()
    logging.info(f'Worker PID: {pid} processing {len(chunk)} importance ...')
    partial_sum = sum(
        get_importance_node(row['pr_val'], row['hc_val']) for row in chunk
    )
    return partial_sum, len(chunk)


def get_average_importance_parallel(
    nodes_path: str, workers: int, chunk_size: int
) -> float:
    pool = mp.Pool(workers)
    reader = csv.DictReader(open(nodes_path, 'r', encoding='utf-8'))
    total_sum = 0.0
    total_count = 0
    for partial_sum, count in tqdm(
        pool.imap_unordered(
            process_importance_chunk, chunked_reader(reader, chunk_size)
        ),
        desc='Avg Importance',
    ):
        total_sum += partial_sum
        total_count += count
    pool.close()
    pool.join()
    return total_sum / total_count


def get_average_importance(chunk_result: List[OutputRow]) -> float:
    total_importance = 0.0
    total_count = 0
    for row in tqdm(chunk_result, desc='Chunk Avg Importance'):
        total_importance += get_importance_node(row['pr_val'], row['hc_val'])
        total_count += 1
    return total_importance / total_count


# ------------------------- Edge Filtering -------------------------


def process_edge_chunk(chunk: List[Dict[str, str]]) -> List[Dict[str, str]]:
    pid = os.getpid()
    logging.info(f'Worker PID: {pid} processing {len(chunk)} edges ...')
    output = []
    for row in chunk:
        if row['src'] in _matched_ids or row['dst'] in _matched_ids:
            output.append(row)
        elif row['src'] in _random_ids or row['dst'] in _random_ids:
            output.append(row)
    return output


def filter_edges_parallel(
    edges_path: str,
    matched_ids: Set[str],
    random_ids: Set[str],
    output_path: str,
    workers: int,
    chunk_size: int,
) -> None:
    logging.info('Filtering 1-hop edges...')

    pool = mp.Pool(
        workers, initializer=init_edge_worker, initargs=(matched_ids, random_ids)
    )
    reader = csv.DictReader(open(edges_path, 'r', encoding='utf-8'))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError('Edge file missing headers.')

    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        total_written = 0
        for chunk_rows in tqdm(
            pool.imap_unordered(process_edge_chunk, chunked_reader(reader, chunk_size)),
            desc='Filtering edges',
        ):
            writer.writerows(chunk_rows)
            total_written += len(chunk_rows)

    pool.close()
    pool.join()
    logging.info(f'Filtered edge file written with {total_written:,} 1-hop edges')


# ------------------------- Node Filtering -------------------------


def process_node_chunk(chunk: List[Dict[str, str]]) -> List[Dict[str, str]]:
    pid = os.getpid()
    logging.info(f'Worker PID: {pid} processing {len(chunk)} nodes ...')
    return [row for row in chunk if row['node_id'] in _keep_ids]


def add_hops_to_node_file(
    scored_node_path: str,
    filtered_edge_path: str,
    output_path: str,
    matched_ids: Set[str],
    random_ids: Set[str],
    workers: int,
    chunk_size: int,
) -> None:
    edge_node_ids = set()
    with open(filtered_edge_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_node_ids.add(row['src'])
            edge_node_ids.add(row['dst'])

    print(f'Edges: {len(edge_node_ids)}')
    print(f'Randoms: {len(random_ids)}')
    keep_ids = matched_ids.union(edge_node_ids, random_ids)
    print(f'Keeps: {len(keep_ids)}')

    unique_to_edges = (
        edge_node_ids
        - edge_node_ids.intersection(matched_ids)
        - edge_node_ids.intersection(random_ids)
    )
    print(f'Unique to Edges: {len(unique_to_edges)}')

    pool = mp.Pool(workers, initializer=init_node_worker, initargs=(keep_ids,))
    reader = csv.DictReader(open(scored_node_path, 'r', encoding='utf-8'))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError('Node file missing headers.')

    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        total_written = 0
        for chunk_rows in tqdm(
            pool.imap_unordered(process_node_chunk, chunked_reader(reader, chunk_size)),
            desc='Filtering Nodes',
        ):
            writer.writerows(chunk_rows)
            total_written += len(chunk_rows)

    pool.close()
    pool.join()
    logging.info(f'One-hop node file written with {total_written:,} nodes')


# ------------------------- Main Merge Function -------------------------


def load_dqr_cache_and_build_trie(dqr_path: str) -> ReversedDomainTrie:
    trie = ReversedDomainTrie()
    with open(dqr_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rev_domain = reverse_domain(row['domain'])
            trie.insert(rev_domain, float(row['pc1']))
    return trie


def write_scored_subset_csv(
    input_path: str,
    output_path_subset: str,
    matched_ids: Set[str],
    random_ids: Set[str],
) -> None:
    """Writes a filtered version of a scored node CSV including only
    matched and random domain IDs.
    """
    os.makedirs(os.path.dirname(output_path_subset), exist_ok=True)

    with (
        open(input_path, 'r', encoding='utf-8') as in_f,
        open(output_path_subset, 'w', newline='', encoding='utf-8') as out_f,
    ):
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError('Input file missing headers')

        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        kept = 0
        for row in tqdm(reader, desc='Writing matched + random rows'):
            node_id = row['node_id']
            if node_id in matched_ids or node_id in random_ids:
                writer.writerow(row)
                kept += 1

    logging.info(f'Wrote {kept:,} matched/random rows to: {output_path_subset}')


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
    total_random = 0
    matched_domain_ids: Set[str] = set()
    random_domain_ids: Set[str] = set()

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

            avg_importance = get_average_importance(result_rows)
            random_count = 0
            max_random = int(matched * 1.0)
            for row in result_rows:
                if row['cr_score'] != -1.0:
                    matched_domain_ids.add(row['node_id'])
                elif random_count < max_random and random.uniform(
                    0, 1
                ) < get_importance_probability_node(
                    row['pr_val'], row['hc_val'], avg_importance
                ):
                    random_count += 1
                    random_domain_ids.add(row['node_id'])
            total_random += random_count

        pool.close()
        pool.join()

    logging.info(f'Matched domains: {total_matched:,}')
    logging.info(f'Unmatched domains: {total_unmatched:,}')
    logging.info(f'Random domains: {total_random}')

    logging.info(f'Constructing subset csv Match + Random')

    filtered_scored_output_subset = output_path.replace('.csv', '_matched_random.csv')
    write_scored_subset_csv(
        input_path=output_path,
        output_path_subset=filtered_scored_output_subset,
        matched_ids=matched_domain_ids,
        random_ids=random_domain_ids,
    )

    logging.info(f'Filtering edges to include only 1-hop neighbors...')
    filter_edges_parallel(
        edges_path=edges_path,
        matched_ids=matched_domain_ids,
        random_ids=random_domain_ids,
        output_path=filtered_edges_output_path,
        workers=workers,
        chunk_size=chunk_size,
    )

    logging.info('Filtering nodes by included edges and matches...')
    added_nodes_output_path = output_path.replace('.csv', '_added_one_hop.csv')
    add_hops_to_node_file(
        scored_node_path=output_path,
        filtered_edge_path=filtered_edges_output_path,
        output_path=added_nodes_output_path,
        matched_ids=matched_domain_ids,
        random_ids=random_domain_ids,
        workers=workers,
        chunk_size=chunk_size,
    )
