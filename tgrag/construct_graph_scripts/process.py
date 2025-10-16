# Processes the graph with temp files / external sort for efficiency:
# - Compute stats on graph
# - Discard nodes below a degree threshold
# - And newly-isolated nodes
# - Add timestamps to vertices and edges (outputs in csv.gz)

import gzip
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from tgrag.utils.data_loading import (
    count_lines,
    gz_line_reader,
    iso_week_to_timestamp,
)


def run_sort(
    in_path: str | Path,
    out_path: str | Path,
    *,
    sort_cmd: str = 'sort',
    mem: str = '60%',
    tmpdir: str | Path,
    delimiter: Optional[str] = None,
    key_start_col: Optional[int] = None,
    key_numeric: bool = False,
    unique: bool = False,
) -> None:
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    cmd = [sort_cmd, '-S', mem, '-T', str(tmpdir)]
    if delimiter is not None:
        cmd += ['-t', delimiter]
    if key_start_col is not None:
        k = f'{key_start_col},{key_start_col}' + ('n' if key_numeric else '')
        cmd += ['-k', k]
    if unique:
        cmd += ['-u']
    cmd += [str(in_path)]
    out_path = Path(out_path)
    with out_path.open('w', encoding='utf-8', newline='') as fout:
        p = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.PIPE, text=True, env=env
        )
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"sort failed: {' '.join(cmd)}\n{err}")


def write_endpoints(edges_gz: Path, endpoints_path: str | Path) -> Tuple[int, int]:
    """Write one endpoint per line (both src and dst). Return (#edges, #lines_written)."""
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


def count_sorted_keys(sorted_path: str | Path, out_tsv: str | Path) -> int:
    """Given a file sorted by key (one key per line), write 'key<TAB>count' and return #unique."""
    uniq = 0
    prev = None
    c = 0
    with (
        open(sorted_path, 'r', encoding='utf-8', newline='') as fin,
        open(out_tsv, 'w', encoding='utf-8', newline='') as fout,
    ):
        for key in fin:
            key = key.rstrip('\n').strip()
            if not key:
                continue
            if prev is None:
                prev = key
                c = 1
            elif key == prev:
                c += 1
            else:
                fout.write(f'{prev}\t{c}\n')
                uniq += 1
                prev = key
                c = 1
        if prev is not None:
            fout.write(f'{prev}\t{c}\n')
            uniq += 1
    return uniq


def filter_domains_by_degree(deg_tsv: str | Path, kept_path: str | Path, k: int) -> int:
    """Write sorted list of domains whose degree > k. Return count kept."""
    kept = 0
    tmp_unsorted = f'{kept_path}.unsorted'
    with (
        open(deg_tsv, 'r', encoding='utf-8', newline='') as fin,
        open(tmp_unsorted, 'w', encoding='utf-8', newline='') as fout,
    ):
        for line in fin:
            dom, deg_str = line.rstrip('\n').split('\t')
            dom = dom.strip()
            if int(deg_str) > k:
                fout.write(f'{dom}\n')
                kept += 1
    with tempfile.TemporaryDirectory(prefix='extsort_k_') as td:
        run_sort(
            tmp_unsorted, kept_path, tmpdir=td, sort_cmd='sort', mem='40%', unique=True
        )
    os.remove(tmp_unsorted)
    return kept


def merge_join_filter_edges(
    edges_in: str | Path,
    kept_sorted: str | Path,
    edges_out: str | Path,
    *,
    by_col: int,
    sort_cmd: str,
    mem: str,
    tmpdir: str | Path,
) -> None:
    """Keep only edges whose key column (1=src, 2=dst) exists in kept_sorted."""
    edges_sorted = Path(tmpdir) / f'edges.sorted.by{by_col}.tsv'
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
        open(kept_sorted, 'r', encoding='utf-8', newline='') as fkeep,
        open(edges_sorted, 'r', encoding='utf-8', newline='') as fedges,
        open(edges_out, 'w', encoding='utf-8', newline='') as fout,
    ):
        k = fkeep.readline().rstrip('\n').strip()
        e = fedges.readline()
        while k and e:
            src, dst = e.rstrip('\n').split('\t')
            src = src.strip()
            dst = dst.strip()
            key = src if by_col == 1 else dst
            if key == k:
                fout.write(f'{src}\t{dst}\n')
                last = key
                while True:
                    pos = fedges.tell()
                    e = fedges.readline()
                    if not e:
                        break
                    src2, dst2 = e.rstrip('\n').split('\t')
                    src2 = src2.strip()
                    dst2 = dst2.strip()
                    key2 = src2 if by_col == 1 else dst2
                    if key2 == last:
                        fout.write(f'{src2}\t{dst2}\n')
                    else:
                        fedges.seek(pos)
                        break
                k = fkeep.readline().rstrip('\n').strip()
            elif key < k:
                e = fedges.readline()
            else:
                k = fkeep.readline().rstrip('\n').strip()
    try:
        os.remove(edges_sorted)
    except FileNotFoundError:
        pass


def _compute_density(V: int, E: int) -> float:
    if V <= 1:
        return 0.0
    return E / (V * (V - 1))


def _compute_vertices_sorted_unique(
    vert_path: str | Path, *, sort_cmd: str, mem: str, tmpdir: Path
) -> Path:
    """Return path to a sorted-unique vertex (one domain per line)."""
    raw = Path(tmpdir) / 'vertex.raw.txt'
    with (
        gzip.open(vert_path, 'rt', encoding='utf-8', newline='') as fin,
        open(raw, 'w', encoding='utf-8', newline='') as fout,
    ):
        for line in fin:
            dom = line.rstrip('\n').strip()
            if dom:
                fout.write(dom + '\n')
    vert_sorted = Path(tmpdir) / 'vertex.sorted.txt'
    run_sort(raw, vert_sorted, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem, unique=True)
    return vert_sorted


def _compute_deg_domains_sorted_unique(
    deg_tsv: str | Path, *, sort_cmd: str, mem: str, tmpdir: Path
) -> Path:
    raw = Path(tmpdir) / 'deg_domains.txt'
    with (
        open(deg_tsv, 'r', encoding='utf-8', newline='') as fin,
        open(raw, 'w', encoding='utf-8', newline='') as fout,
    ):
        for line in fin:
            dom, _ = line.rstrip('\n').split('\t')
            dom = dom.strip()
            fout.write(dom + '\n')
    sorted_path = Path(tmpdir) / 'deg_domains.sorted.txt'
    run_sort(raw, sorted_path, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem, unique=True)
    return sorted_path


def _stats_initial(
    deg_tsv: str | Path,
    E: int,
    vert_path: str | Path,
    *,
    sort_cmd: str,
    mem: str,
    tmpdir: Path,
) -> dict:
    r"""Computes:
    - isolated = |V \ deg_domains|
    - leaves = count(deg==1) from deg_tsv
    - min_deg = 0 if isolated>0 else min(deg_tsv)
    - mean_deg = sum(deg_tsv) / V.
    """
    V_deg = 0
    sum_deg = 0
    min_deg = None
    max_deg = 0
    leaves = 0
    with open(deg_tsv, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if not line:
                continue
            dom, dstr = line.rstrip('\n').split('\t')
            dom = dom.strip()
            if not dom:
                continue
            d = int(dstr)
            V_deg += 1
            sum_deg += d
            if min_deg is None or d < min_deg:
                min_deg = d
            if d > max_deg:
                max_deg = d
            if d == 1:
                leaves += 1

    # vert size (unique) and isolated via set-diff
    vert_sorted = _compute_vertices_sorted_unique(
        vert_path, sort_cmd=sort_cmd, mem=mem, tmpdir=tmpdir
    )
    deg_sorted = _compute_deg_domains_sorted_unique(
        deg_tsv, sort_cmd=sort_cmd, mem=mem, tmpdir=tmpdir
    )

    V_vert = count_lines(str(vert_sorted))  # unique count
    # stream set difference: vert \ deg
    isolated = 0
    with (
        open(vert_sorted, 'r', encoding='utf-8') as A,
        open(deg_sorted, 'r', encoding='utf-8') as B,
    ):
        a = A.readline()
        b = B.readline()
        while a:
            da = a.rstrip('\n')
            if b:
                db = b.rstrip('\n')
                if da == db:
                    a = A.readline()
                    b = B.readline()
                elif da < db:
                    isolated += 1
                    a = A.readline()
                else:
                    b = B.readline()
            else:
                isolated += 1
                a = A.readline()

    # degree min/mean using init size
    deg_min = 0 if isolated > 0 else (min_deg if min_deg is not None else 0)
    mean_deg = (sum_deg / V_vert) if V_vert else 0.0
    density = _compute_density(V_vert, E)

    print(
        f'[STATS:initial] V={V_vert:,}  E={E:,}  deg(min/mean/max)={deg_min}/{mean_deg:.3f}/{max_deg}  '
        f'isolated={isolated:,}  leaves={leaves:,}  density={density:.6g}'
    )
    return dict(
        V=V_vert,
        E=E,
        min=deg_min,
        mean=mean_deg,
        max=max_deg,
        isolated=isolated,
        leaves=leaves,
        density=density,
    )


def recompute_vertices_from_edges(
    edges_path: str | Path,
    out_vertices_gz: str | Path,
    ts_str: str,
    *,
    sort_cmd: str,
    mem: str,
) -> dict:
    """Final stats + write vertices.csv.gz (domain,ts) for nodes with degree > 0 in filtered graph."""
    with tempfile.TemporaryDirectory(prefix='deg_') as tdf:
        td = Path(tdf)
        endpoints = td / 'endpoints.txt'
        with (
            open(edges_path, 'r', encoding='utf-8', newline='') as fin,
            open(endpoints, 'w', encoding='utf-8', newline='') as fout,
        ):
            for line in fin:
                if not line:
                    continue
                s, t = line.rstrip('\n').split('\t')
                s = s.strip()
                t = t.strip()
                if not s or not t:
                    continue
                fout.write(f'{s}\n')
                fout.write(f'{t}\n')
        sorted_endpoints = td / 'endpoints.sorted.txt'
        run_sort(endpoints, sorted_endpoints, tmpdir=td, sort_cmd=sort_cmd, mem=mem)

        # stream degrees + write vertices
        V = 0
        sum_deg = 0
        min_deg = None
        max_deg = 0
        leaves = 0
        prev = None
        c = 0
        with (
            open(sorted_endpoints, 'r', encoding='utf-8', newline='') as fin,
            gzip.open(out_vertices_gz, 'wt', encoding='utf-8', newline='') as gzv,
        ):
            gzv.write('domain,ts\n')
            for dom in fin:
                dom = dom.rstrip('\n').strip()
                if not dom:
                    continue
                if prev is None:
                    prev, c = dom, 1
                elif dom == prev:
                    c += 1
                else:
                    V += 1
                    sum_deg += c
                    if min_deg is None or c < min_deg:
                        min_deg = c
                    if c > max_deg:
                        max_deg = c
                    if c == 1:
                        leaves += 1
                    gzv.write(f'{prev},{ts_str}\n')
                    prev, c = dom, 1
            if prev is not None:
                V += 1
                sum_deg += c
                if min_deg is None or c < min_deg:
                    min_deg = c
                if c > max_deg:
                    max_deg = c
                if c == 1:
                    leaves += 1
                gzv.write(f'{prev},{ts_str}\n')

        E = count_lines(str(edges_path))
        mean_deg = (sum_deg / V) if V else 0.0
        isolated = 0  # final graph excludes deg==0 by construction
        density = _compute_density(V, E)
        print(
            f'[STATS:final] V={V:,}  E={E:,}  deg(min/mean/max)={min_deg}/{mean_deg:.3f}/{max_deg}  '
            f'isolated={isolated:,}  leaves={leaves:,}  density={density:.6g}'
        )
        return dict(
            V=V,
            E=E,
            min=min_deg,
            mean=mean_deg,
            max=max_deg,
            isolated=isolated,
            leaves=leaves,
            density=density,
        )


def process_graph(
    graph: str,
    slice_str: str,
    min_deg: int,
    mem: str = '60%',
) -> None:
    graph_path = Path(graph)
    e_gz = graph_path / 'edges.txt.gz'
    sort_cmd = 'sort'

    ts = iso_week_to_timestamp(slice_str)  # e.g., CC-MAIN-2024-18 -> 20240429

    p = graph_path / 'vertices.txt.gz'
    vertices = p

    out_dir = graph_path
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='extsort_') as tdf:
        td = Path(tdf)

        # (1) degrees from original graph
        print('[STEP] computing initial degrees')
        endpoints = td / 'endpoints.txt'
        E_init, _ = write_endpoints(e_gz, endpoints)
        endpoints_sorted = td / 'endpoints.sorted.txt'
        run_sort(endpoints, endpoints_sorted, tmpdir=td, sort_cmd=sort_cmd, mem=mem)
        degrees_initial = td / 'degrees.initial.tsv'
        count_sorted_keys(endpoints_sorted, degrees_initial)

        # initial stats
        _stats_initial(
            degrees_initial, E_init, vertices, sort_cmd=sort_cmd, mem=mem, tmpdir=td
        )

        # (2) discard degree <= K
        kept_domains_sorted = td / 'kept_domains.sorted.txt'
        kept_V = filter_domains_by_degree(degrees_initial, kept_domains_sorted, min_deg)
        print(f'[INFO] kept domains (deg>{min_deg}): {kept_V:,}')

        # (3) filter edges where both endpoints kept (2 merge-joins)
        print('[STEP] filtering edges (src∈kept AND dst∈kept)')
        edges_all = td / 'edges.tsv'
        with open(edges_all, 'w', encoding='utf-8', newline='') as fout:
            for line in gz_line_reader(e_gz):
                if not line:
                    continue
                try:
                    src, dst = line.split('\t', 1)
                except ValueError:
                    continue
                src = src.strip()
                dst = dst.strip()
                if not src or not dst:
                    continue
                fout.write(f'{src}\t{dst}\n')

        edges_src_kept = td / 'edges.src_kept.tsv'
        merge_join_filter_edges(
            edges_all,
            kept_domains_sorted,
            edges_src_kept,
            by_col=1,
            sort_cmd=sort_cmd,
            mem=mem,
            tmpdir=td,
        )

        filtered_edges_tsv = td / 'edges.filtered.tsv'
        merge_join_filter_edges(
            edges_src_kept,
            kept_domains_sorted,
            filtered_edges_tsv,
            by_col=2,
            sort_cmd=sort_cmd,
            mem=mem,
            tmpdir=td,
        )

        # dedupe edges
        edges_dedup_tsv = td / 'edges.filtered.dedup.tsv'
        run_sort(
            filtered_edges_tsv,
            edges_dedup_tsv,
            tmpdir=td,
            sort_cmd=sort_cmd,
            mem=mem,
            unique=True,
        )
        print(f'[STATS] edges after dst-join: {count_lines(str(filtered_edges_tsv)):,}')
        print(f'[STATS] edges after dedup:    {count_lines(str(edges_dedup_tsv)):,}')
        filtered_edges_tsv = edges_dedup_tsv

        # & write final edges.csv.gz with ts's
        edges_csv_gz = out_dir / 'edges.csv.gz'
        with (
            gzip.open(edges_csv_gz, 'wt', encoding='utf-8', newline='') as gzout,
            open(filtered_edges_tsv, 'r', encoding='utf-8', newline='') as fin,
        ):
            gzout.write('src,dst,ts\n')
            for line in fin:
                src, dst = line.rstrip('\n').split('\t')
                src = src.strip()
                dst = dst.strip()
                if not src or not dst:
                    continue
                gzout.write(f'{src},{dst},{ts}\n')

        # recompute degrees on filtered graph, write vertices.csv.gz, & log final stats
        print('[STEP] recomputing degrees on filtered graph + writing vertices')
        vertices_csv_gz = out_dir / 'vertices.csv.gz'
        _ = recompute_vertices_from_edges(
            filtered_edges_tsv, vertices_csv_gz, ts, sort_cmd=sort_cmd, mem=mem
        )

        print(f'[DONE] edges -> {edges_csv_gz}')
        print(f'[DONE] vertices -> {vertices_csv_gz}')
