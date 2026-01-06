import csv
import gzip
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import IO, Callable, Dict, Iterator, List, Set, Tuple

from tqdm import tqdm


def check_processed_file(processed: Path) -> None:
    processed_count = 0
    label_counts = {0: 0, 1: 0}
    headers = None

    with processed.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        for row in reader:
            if not row:
                continue
            processed_count += 1
            label = int(row[1])
            if label in label_counts:
                label_counts[label] += 1

    print('Processed: rows:', processed_count)
    print('Headers:', headers)
    print('Label counts:')
    print('  0:', label_counts[0])
    print('  1:', label_counts[1])


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def count_lines(path: str) -> int:
    if path.endswith('.gz'):
        c = 0
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
            for _ in f:
                c += 1
        return c
    else:
        out = run(['wc', '-l', path]).stdout.strip().split()[0]
        return int(out)


def gz_line_reader(path: str | Path) -> Iterator[str]:
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        for line in f:
            yield line.rstrip('\n')


def read_vertex_file(path: str) -> Set[str]:
    result: Set[str] = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            result.add(line.strip())
    return result


def extract_all_domains(vertices_gz: str, edges_gz: str, out_txt: str) -> None:
    with open(out_txt, 'w', encoding='utf-8', newline='') as out:
        for line in gz_line_reader(vertices_gz):
            dom = line.strip()
            if dom:
                out.write(dom + '\n')
        for line in gz_line_reader(edges_gz):
            if not line:
                continue
            try:
                src, dst = line.split('\t', 1)
            except ValueError:
                continue
            src = src.strip()
            dst = dst.strip()
            if src:
                out.write(src + '\n')
            if dst:
                out.write(dst + '\n')


def read_edge_file(path: str, id_to_domain: Dict[int, str]) -> Set[Tuple[str, str]]:
    result: Set[Tuple[str, str]] = set()
    get = id_to_domain.get
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            src_id = int(parts[0])
            dst_id = int(parts[1])
            src = get(src_id)
            dst = get(dst_id)
            if src is None or dst is None:
                continue  # skip if an endpoint wasn't present in vertices map
            result.add((src, dst))
    return result


def open_file(path: str) -> Callable[..., IO]:
    return gzip.open if path.endswith('.gz') else open


def load_edges(edge_file: str) -> List[Tuple[str, str]]:
    edges = []
    with open_file(edge_file)(edge_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading edge file'):
            parts = line.strip().split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


def load_node_domain_map(node_file: str) -> Tuple[dict, dict]:
    id_to_domain = {}
    domain_to_id = {}
    with open_file(node_file)(node_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading node file'):
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, domain = parts
                id_to_domain[node_id] = domain
                domain_to_id[domain] = node_id
    return id_to_domain, domain_to_id


def write_endpoints(edges_gz: Path, endpoints_path: str | Path) -> Tuple[int, int]:
    """
    Write one endpoint per line (both src and dst). 
    Return (#edges, #lines_written).
    """
    E = 0
    lines = 0
    with open(endpoints_path, 'w', encoding='utf-8', newline='') as out:
        for line in gz_line_reader(edges_gz):
            src, dst = line.split('\t', 1)
            src = src.strip()
            dst = dst.strip()
            out.write(f'{src}\n')
            out.write(f'{dst}\n')
            lines += 2
            E += 1
    return E, lines
