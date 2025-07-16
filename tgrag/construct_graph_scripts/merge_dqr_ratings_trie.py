import csv
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, Generator, Iterable, List, Tuple, TypedDict, cast

from tqdm import tqdm

from tgrag.utils.matching import reverse_domain


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
    output_path: str,
    workers: int = 16,
    chunk_size: int = 100_000,
) -> None:
    trie = load_dqr_cache_and_build_trie(dqr_path)
    total_matched = 0
    total_unmatched = 0

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

        pool.close()
        pool.join()

    logging.info(f'\nMatching Complete:')
    logging.info(f'Matched domains   : {total_matched:,}')
    logging.info(f'Unmatched domains : {total_unmatched:,}')
