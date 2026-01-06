# File writers.
# Include writers and loaders used for graphs at various steps and label datasets.

import csv
import gzip
import os
import tempfile
from pathlib import Path
from typing import Tuple

from tgrag.utils.analytics import (
    compute_density,
    count_lines,
    count_sorted_keys,
)
from tgrag.utils.io import run_ext_sort
from tgrag.utils.readers import line_reader


# For graphs
# ----------
def write_endpoints(edges_gz: Path, endpoints_path: str | Path) -> Tuple[int, int]:
    """Write all edge endpoints to a text file, one per line.

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
        for line in line_reader(edges_gz):
            src, dst = line.split('\t', 1)
            src = src.strip()
            dst = dst.strip()
            out.write(f'{src}\n')
            out.write(f'{dst}\n')
            lines += 2
            E += 1
    return E, lines


def build_from_BCC(
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
        run_ext_sort(src_path, src_sorted, tmpdir=td, sort_cmd=sort_cmd, mem=mem)
        run_ext_sort(dst_path, dst_sorted, tmpdir=td, sort_cmd=sort_cmd, mem=mem)
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


# For domain labels
# ------------------


def write_aggr_labelled(
    domain_labels: dict[str, list[float]], output_csv: Path
) -> None:
    """Write merged domain labels to a CSV file.

    For each domain, compute the average of its associated labels and assign a
    final binary label: 1 if the average is at least 0.5, otherwise 0. The output
    CSV contains one row per domain with columns "domain" and "label".

    Parameters:
        domain_labels : dict
            Mapping from domain to a list of numeric labels.
        output_csv : pathlib.Path
            Path to the output CSV file.

    Returns:
        None
    """
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in sorted(domain_labels.items()):
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            final_label = 1 if avg_label >= 0.5 else 0
            writer.writerow([domain, final_label])
