import gzip
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from tgrag.utils.data_io import count_lines, write_endpoints


def compute_density(V: int, E: int) -> float:
    if V <= 1:
        return 0.0
    return E / (V * (V - 1))


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
    """Sort a text file using the external Unix `sort` command and write the result
    to an output file.

    Parameters:
        in_path : str or pathlib.Path
            Path to input vertex file.
        out_path : str or pathlib.Path
            Path where the sorted output will be written.
        sort_cmd : str, optional
            Name or path of the `sort` executable to run (default: ``"sort"``).
        mem : str, optional
            Memory limit passed to `sort` via ``-S`` (e.g. ``"60%"``).
        tmpdir : str or pathlib.Path
            Directory used by `sort` for temporary files.
        delimiter : str, optional
            Field delimiter to use for sorting. If ``None``, the comma delimiter is used.
        key_start_col : int, optional
            1-based index of the field to sort on. If ``None``, the entire line is
            used as the sort key.
        key_numeric : bool, optional
            If ``True``, perform a numeric sort on the selected key.
        unique : bool, optional
            If ``True``, suppress duplicate lines in the output (adds ``-u``).

    Returns: None

    Raises:
        RuntimeError
            If the `sort` command exits with a non-zero status. The error output from
            `sort` is included in the exception message.
    """
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    cmd = [sort_cmd, '-S', mem, '-T', str(tmpdir)]
    if delimiter is None:
        delimiter = ','
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
    run_sort(endpoints, endpoints_sorted, tmpdir=tmpdir, sort_cmd=sort_cmd, mem=mem)

    degrees = tmpdir / 'degrees.initial.tsv'
    count_sorted_keys(endpoints_sorted, degrees)

    endpoints.unlink()
    endpoints_sorted.unlink()
    return degrees, E


def compute_vertices_from_edges(
    edges_path: str | Path,
    out_vertices_gz: str | Path,
    ts_str: str,
    *,
    sort_cmd: str = 'sort',
    mem: str = '60%',
) -> dict:
    """Writes the graph's biggest connected component vertex files, with in- and out-degrees.

    Output schema:
        domain,ts,in_deg,out_deg

    Uses external sort to compute in-degree and out-degree separately and then
    merge-joins the two degree tables in a scalable fashion.
    """
    with tempfile.TemporaryDirectory(prefix='deg_') as tdf:
        td = Path(tdf)

        src_path = td / 'src.txt'
        dst_path = td / 'dst.txt'
        with (
            open(edges_path, 'r', encoding='utf-8', newline='') as fin,
            open(src_path, 'w', encoding='utf-8', newline='') as fsrc,
            open(dst_path, 'w', encoding='utf-8', newline='') as fdst,
        ):
            for line in fin:
                if not line:
                    continue
                try:
                    s, t = line.rstrip('\n').split('\t')
                except ValueError:
                    continue
                s = s.strip()
                t = t.strip()
                if not s or not t:
                    continue
                fsrc.write(f'{s}\n')
                fdst.write(f'{t}\n')

        src_sorted = td / 'src.sorted.txt'
        dst_sorted = td / 'dst.sorted.txt'
        run_sort(src_path, src_sorted, tmpdir=td, sort_cmd=sort_cmd, mem=mem)
        run_sort(dst_path, dst_sorted, tmpdir=td, sort_cmd=sort_cmd, mem=mem)
        os.remove(src_path)
        os.remove(dst_path)

        out_deg_tsv = td / 'out_deg.tsv'
        in_deg_tsv = td / 'in_deg.tsv'
        count_sorted_keys(src_sorted, out_deg_tsv)  # domain \t out_deg
        count_sorted_keys(dst_sorted, in_deg_tsv)  # domain \t in_deg
        os.remove(src_sorted)
        os.remove(dst_sorted)

        V = 0
        sum_deg = 0
        min_deg = None
        max_deg = 0
        leaves = 0

        with (
            open(in_deg_tsv, 'r', encoding='utf-8', newline='') as fin_in,
            open(out_deg_tsv, 'r', encoding='utf-8', newline='') as fin_out,
            gzip.open(out_vertices_gz, 'wt', encoding='utf-8', newline='') as gzv,
        ):
            gzv.write('domain,ts,in_deg,out_deg\n')

            line_in = fin_in.readline()
            line_out = fin_out.readline()

            while line_in or line_out:
                if line_in:
                    d_in, din_str = line_in.rstrip('\n').split('\t')
                    d_in = d_in.strip()
                    din = int(din_str)
                else:
                    d_in = None
                    din = 0

                if line_out:
                    d_out, dout_str = line_out.rstrip('\n').split('\t')
                    d_out = d_out.strip()
                    dout = int(dout_str)
                else:
                    d_out = None
                    dout = 0

                if d_in is not None and (d_out is None or d_in < d_out):
                    dom = d_in
                    indeg = din
                    outdeg = 0
                    line_in = fin_in.readline()
                elif d_out is not None and (d_in is None or d_out < d_in):
                    dom = d_out
                    indeg = 0
                    outdeg = dout
                    line_out = fin_out.readline()
                else:
                    # Same domain present in both tables
                    dom = d_in  # == d_out
                    indeg = din
                    outdeg = dout
                    line_in = fin_in.readline()
                    line_out = fin_out.readline()

                if not dom:
                    continue

                deg_total = indeg + outdeg
                V += 1
                sum_deg += deg_total
                if min_deg is None or deg_total < min_deg:
                    min_deg = deg_total
                if deg_total > max_deg:
                    max_deg = deg_total
                if deg_total == 1:
                    leaves += 1

                gzv.write(f'{dom},{ts_str},{indeg},{outdeg}\n')

        os.remove(in_deg_tsv)
        os.remove(out_deg_tsv)

        E = count_lines(str(edges_path))
        mean_deg = (sum_deg / V) if V else 0.0
        isolated = 0  # by construction
        density = compute_density(V, E)

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
