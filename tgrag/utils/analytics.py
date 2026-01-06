# Functions for statistics and computing metrics

import gzip
from pathlib import Path
from typing import Optional, Tuple

from tgrag.utils.io import run, run_ext_sort
from tgrag.utils.writers import write_endpoints


def compute_density(V: int, E: int) -> float:
    """Compute the directed graph density.

    The density is defined as E / (V * (V - 1)), corresponding to the fraction of
    all possible directed edges that are present. If V <= 1, the density is defined
    as 0.0.

    Parameters:
        V : int
            Number of vertices in the graph.
        E : int
            Number of edges in the graph.

    Returns:
        float
            Graph density in the range [0.0, 1.0].
    """
    if V <= 1:
        return 0.0
    return E / (V * (V - 1))


def count_lines(path: str) -> int:
    """Count the number of lines in a text file, supporting gzip-compressed files.

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


def stats(
    deg_tsv: Path, E: int, vert_path: Path, *, sort_cmd: str, mem: str, tmpdir: Path
) -> None:
    """Compute and print initial graph statistics.

    Parameters:
        deg_tsv : Path
            Path to degree TSV file.
        E : int
            Number of edges in the graph.
        vert_path : Path
            Path to vertex list file.
        sort_cmd : str
            Sort executable to use.
        mem : str
            Memory limit passed to the sort command.
        tmpdir : Path
            Temporary directory for intermediate files.
    """
    V_deg = sum_deg = leaves = 0
    min_deg: Optional[int] = None
    max_deg = 0

    with open(deg_tsv) as f:
        for line in f:
            dom, d_str = line.strip().split('\t')
            d = int(d_str)
            V_deg += 1
            sum_deg += d
            min_deg = d if min_deg is None else min(min_deg, d)
            max_deg = max(max_deg, d)
            if d == 1:
                leaves += 1

    # sorted unique vertices
    raw = tmpdir / 'vertex.raw.txt'
    with gzip.open(vert_path, 'rt') as fin, open(raw, 'w') as fout:
        for line in fin:
            dom = line.strip()
            if dom:
                fout.write(dom + '\n')

    vert_sorted = tmpdir / 'vertex.sorted.txt'
    run_ext_sort(
        raw, vert_sorted, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem, unique=True
    )

    V = count_lines(str(vert_sorted))

    deg_min = 0 if V_deg < V else min_deg or 0
    mean_deg = sum_deg / V if V else 0
    density = compute_density(V, E)

    print(
        f'[STATS] V={V:,}  E={E:,}  deg(min/mean/max)={deg_min}/{mean_deg:.3f}/{max_deg}  leaves={leaves:,}  density={density:.6g}'
    )


def count_sorted_keys(sorted_path: Path, out_tsv: Path) -> int:
    """Count occurrences of consecutive identical keys in a sorted file.

    Parameters:
        sorted_path : Path
            Path to a file containing sorted keys, one per line.
        out_tsv : Path
            Path to the output TSV file to write "key<TAB>count" lines.

    Returns:
        int
            Number of unique keys written to the output file.
    """
    uniq = 0
    prev = None
    c = 0
    with open(sorted_path) as fin, open(out_tsv, 'w') as fout:
        for key in fin:
            key = key.rstrip('\n').strip()
            if not key:
                continue
            if prev is None:
                prev, c = key, 1
            elif key == prev:
                c += 1
            else:
                fout.write(f'{prev}\t{c}\n')
                uniq += 1
                prev, c = key, 1
        if prev is not None:
            fout.write(f'{prev}\t{c}\n')
            uniq += 1
    return uniq


def compute_degrees(
    edges_gz: Path, tmpdir: Path, *, sort_cmd: str, mem: str
) -> Tuple[Path, int]:
    """Compute initial node degrees from an edge list.

    Parameters:
        edges_gz : Path
            Path to a gzip-compressed edge file ("src<TAB>dst" per line).
        tmpdir : Path
            Temporary directory to store intermediate files.
        sort_cmd : str
            Sort executable to use.
        mem : str
            Memory limit passed to the sort command.

    Returns:
        (Path, int)
            Path to the degree TSV file and the number of edges processed.
    """
    endpoints = tmpdir / 'endpoints.txt'
    E, _ = write_endpoints(edges_gz, endpoints)

    endpoints_sorted = tmpdir / 'endpoints.sorted.txt'
    run_ext_sort(endpoints, endpoints_sorted, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem)

    degrees = tmpdir / 'degrees.initial.tsv'
    count_sorted_keys(endpoints_sorted, degrees)

    endpoints.unlink()
    endpoints_sorted.unlink()
    return degrees, E
