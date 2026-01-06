# Processes the graph with temp files / external sort for efficiency:
# - Compute stats on graph
# - Discard nodes below a degree threshold
# - And newly-isolated nodes
# - Add timestamps to vertices and edges (outputs in csv.gz)

import gzip
import tempfile
from pathlib import Path
from typing import Optional

from tgrag.utils.data_io import count_lines, gz_line_reader
from tgrag.utils.graph_manip import (
    compute_degrees,
    compute_density,
    compute_vertices_from_edges,
    run_sort,
)
from tgrag.utils.temporal_utils import iso_week_to_timestamp


def filter_domains_by_degree(
    deg_tsv: Path, kept_sorted: Path, k: int, *, sort_cmd: str, mem: str
) -> int:
    """Filter domains by minimum degree threshold.

    Parameters:
        deg_tsv : Path
            Path to the "domain<TAB>degree" TSV file.
        kept_sorted : Path
            Path where the sorted list of kept domains will be written.
        k : int
            Minimum degree threshold (strictly greater than k is kept).
        sort_cmd : str
            Sort executable to use.
        mem : str
            Memory limit passed to the sort command.

    Returns:
        int
            Number of domains kept.
    """
    tmp_unsorted = kept_sorted.with_suffix('.unsorted')
    kept = 0
    with open(deg_tsv) as fin, open(tmp_unsorted, 'w') as fout:
        for line in fin:
            dom, d = line.strip().split('\t')
            if int(d) > k:
                fout.write(dom + '\n')
                kept += 1

    with tempfile.TemporaryDirectory(prefix='extsort_k_') as td:
        run_sort(
            tmp_unsorted,
            kept_sorted,
            tmpdir=td,
            sort_cmd=sort_cmd,
            mem='40%',
            unique=True,
        )

    tmp_unsorted.unlink()
    return kept


def merge_join_filter_edges(
    edges_in: Path,
    kept_sorted: Path,
    edges_out: Path,
    *,
    by_col: int,
    sort_cmd: str,
    mem: str,
    tmpdir: Path,
) -> None:
    """Filter edges by keeping only those whose endpoint appears in a kept-domain list.

    Parameters:
        edges_in : Path
            Path to input edge TSV file.
        kept_sorted : Path
            Path to sorted list of domains to keep.
        edges_out : Path
            Path to output filtered edge file.
        by_col : int
            Column to match on (1 for src, 2 for dst).
        sort_cmd : str
            Sort executable to use.
        mem : str
            Memory limit passed to the sort command.
        tmpdir : Path
            Temporary directory for intermediate files.
    """
    edges_sorted = tmpdir / f'edges.sorted.by{by_col}.tsv'
    run_sort(
        edges_in,
        edges_sorted,
        tmpdir=tmpdir,
        sort_cmd=sort_cmd,
        mem=mem,
        delimiter='\t',
        key_start_col=by_col,
    )

    with (
        open(kept_sorted) as fkeep,
        open(edges_sorted) as fedges,
        open(edges_out, 'w') as fout,
    ):
        k = fkeep.readline().strip()
        e = fedges.readline()
        while k and e:
            src, dst = e.strip().split('\t')
            key = src if by_col == 1 else dst
            if key == k:
                fout.write(f'{src}\t{dst}\n')
                last = key
                while True:
                    pos = fedges.tell()
                    e = fedges.readline()
                    if not e:
                        break
                    src2, dst2 = e.strip().split('\t')
                    key2 = src2 if by_col == 1 else dst2
                    if key2 == last:
                        fout.write(f'{src2}\t{dst2}\n')
                    else:
                        fedges.seek(pos)
                        break
                k = fkeep.readline().strip()
            elif key < k:
                e = fedges.readline()
            else:
                k = fkeep.readline().strip()

    edges_sorted.unlink(missing_ok=True)


def _stats_initial(
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
    run_sort(raw, vert_sorted, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem, unique=True)

    V = count_lines(str(vert_sorted))

    deg_min = 0 if V_deg < V else min_deg or 0
    mean_deg = sum_deg / V if V else 0
    density = compute_density(V, E)

    print(
        f'[STATS:initial] V={V:,}  E={E:,}  deg(min/mean/max)={deg_min}/{mean_deg:.3f}/{max_deg}  leaves={leaves:,}  density={density:.6g}'
    )


def process_graph(graph: str, slice_str: str, min_deg: int, mem: str = '60%') -> None:
    """Process a graph with external sorting and filtering.

    Parameters:
        graph : str
            Path to the graph directory containing edges.txt.gz and vertices.txt.gz.
        slice_str : str
            CC slice identifier (e.g., "CC-MAIN-2024-18").
        min_deg : int
            Minimum degree threshold for keeping nodes.
        mem : str, optional
            Memory limit passed to external sort commands.
    """
    graph_path = Path(graph)
    edges_gz = graph_path / 'edges.txt.gz'
    vertices_gz = graph_path / 'vertices.txt.gz'
    sort_cmd = 'sort'
    ts = iso_week_to_timestamp(slice_str)

    with tempfile.TemporaryDirectory(prefix='extsort_') as tdf:
        td = Path(tdf)

        print('[STEP] computing initial degrees')
        deg_tsv, E = compute_degrees(edges_gz, td, sort_cmd=sort_cmd, mem=mem)

        _stats_initial(deg_tsv, E, vertices_gz, sort_cmd=sort_cmd, mem=mem, tmpdir=td)

        kept_sorted = td / 'kept_domains.sorted.txt'
        kept = filter_domains_by_degree(
            deg_tsv, kept_sorted, min_deg, sort_cmd=sort_cmd, mem=mem
        )
        print(f'[INFO] kept domains (deg>{min_deg}): {kept:,}')

        print('[STEP] filtering edges')
        edges_all = td / 'edges.tsv'
        with open(edges_all, 'w') as fout:
            for line in gz_line_reader(edges_gz):
                if line:
                    try:
                        src, dst = map(str.strip, line.split('\t', 1))
                        if src and dst:
                            fout.write(f'{src}\t{dst}\n')
                    except ValueError:
                        pass

        edges_src_kept = td / 'edges.src_kept.tsv'
        merge_join_filter_edges(
            edges_all,
            kept_sorted,
            edges_src_kept,
            by_col=1,
            sort_cmd=sort_cmd,
            mem=mem,
            tmpdir=td,
        )

        edges_filtered = td / 'edges.filtered.tsv'
        merge_join_filter_edges(
            edges_src_kept,
            kept_sorted,
            edges_filtered,
            by_col=2,
            sort_cmd=sort_cmd,
            mem=mem,
            tmpdir=td,
        )

        edges_dedup = td / 'edges.filtered.dedup.tsv'
        run_sort(
            edges_filtered,
            edges_dedup,
            tmpdir=td,
            sort_cmd=sort_cmd,
            mem=mem,
            unique=True,
        )

        edges_csv = graph_path / 'edges.csv.gz'
        with gzip.open(edges_csv, 'wt') as gzout, open(edges_dedup) as fin:
            gzout.write('src,dst,ts\n')
            for line in fin:
                src, dst = line.strip().split('\t')
                gzout.write(f'{src},{dst},{ts}\n')

        print('[STEP] recomputing degrees and writing vertices')
        vertices_csv = graph_path / 'vertices.csv.gz'
        compute_vertices_from_edges(
            edges_dedup, vertices_csv, ts, sort_cmd=sort_cmd, mem=mem
        )

        print(f'[DONE] edges -> {edges_csv}')
        print(f'[DONE] vertices -> {vertices_csv}')
