import csv
import gzip
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import IO, Callable, Dict, Iterator, List, Set, Tuple

from tqdm import tqdm


def check_processed_file(processed: Path) -> None:
    """
    Inspect a processed CSV file and print basic statistics.

    Prints:
      - Total number of non-empty data rows
      - The header row (if present)
      - Counts of label 0 and label 1

    Parameters:
        processed : pathlib.Path
            Path to the processed CSV file.

    Returns: None
    """
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
    """
    Run a subprocess command and capture its stdout and stderr as text.

    Parameters:
        cmd : list of str
            Command and arguments to execute.
        check : bool, optional
            If True, raise RuntimeError when the command fails (default: True).

    Returns:
        subprocess.CompletedProcess[str]
            The completed process object containing stdout, stderr, and return code.

    Raises:
        RuntimeError
            If `check` is True and the command exits with a non-zero status.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def count_lines(path: str) -> int:
    """
    Count the number of lines in a text file, supporting gzip-compressed files.

    Parameters:
        path : str
            Path to the file.

    Returns:
        int
            Number of lines in the file.
    """
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
    """
    Yield lines from a gzip-compressed text file without trailing newlines.

    Parameters:
        path : str or pathlib.Path
            Path to the gzip-compressed file.

    Yields:
        str
            Each line from the file with the trailing newline removed.
    """
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        for line in f:
            yield line.rstrip('\n')


def read_vertex_file(path: str) -> Set[str]:
    """
    Read a gzip-compressed vertex file into a set of strings.

    Parameters:
        path : str
            Path to the gzip-compressed vertex file.

    Returns:
        set of str
            All unique vertex identifiers found in the file.
    """
    result: Set[str] = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            result.add(line.strip())
    return result


def extract_all_domains(vertices_gz: str, edges_gz: str, out_txt: str) -> None:
    """
    Extract all domain names from vertex and edge files into a single text file.

    Parameters:
        vertices_gz : str
            Path to the gzip-compressed vertex file.
        edges_gz : str
            Path to the gzip-compressed edge file (tab-separated src and dst).
        out_txt : str
            Path to the output text file.

    Returns:
        None
    """
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
    """
    Read a gzip-compressed edge file and map numeric IDs to domain strings.

    Parameters:
        path : str
            Path to the gzip-compressed edge file.
        id_to_domain : dict[int, str]
            Mapping from numeric node IDs to domain names.

    Returns:
        set of (str, str)
            Set of (source_domain, destination_domain) tuples.
    """
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
    """
    Return the appropriate open function for a path.

    Parameters:
        path : str
            Path to the file.

    Returns:
        callable
            A file-opening function compatible with open()'s signature.
    """
    return gzip.open if path.endswith('.gz') else open


def load_edges(path: str) -> List[Tuple[str, str]]:
    """
    Load an edge list from a text or gzip-compressed file, expected to have (source, domain) per line.

    Parameters:
        path : str
            Path to the edge file.

    Returns:
        list of (str, str)
            List of edge tuples.
    """
    edges = []
    with open_file(path)(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading edge file'):
            parts = line.strip().split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


def load_node_domain_map(path: str) -> Tuple[dict, dict]:
    """
    Load a node-to-domain and domain-to-node mapping from a file.

    Parameters:
        path : str
            Path to the node file, expected to have (node_id, domain) per line.

    Returns:
        (dict, dict)
            A tuple (id_to_domain, domain_to_id).
    """
    id_to_domain = {}
    domain_to_id = {}
    with open_file(path)(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading node file'):
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, domain = parts
                id_to_domain[node_id] = domain
                domain_to_id[domain] = node_id
    return id_to_domain, domain_to_id


def write_endpoints(edges_gz: Path, endpoints_path: str | Path) -> Tuple[int, int]:
    """
    Write all edge endpoints to a text file, one per line.

    Parameters:
        edges_gz : pathlib.Path
            Path to the gzip-compressed edge file.
        endpoints_path : str or pathlib.Path
            Path to the output file.

    Returns:
        (int, int)
            A tuple (#edges_read, #lines_written).
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
