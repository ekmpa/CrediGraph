# Processes the graph with temp files / external sort for efficiency:
# - Assign numeric IDs to nodes
# - Compute stats on graph
# - Discard nodes below a degree threshold
# - Add timestamps to vertices and edges (outputs in csv.gz)
# - Generate target labels for nodes based on DQR data

import csv
import gzip
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np

from tgrag.utils.data_loading import (
    count_lines,
    extract_all_domains,
    gz_line_reader,
    iso_week_to_timestamp,
)
from tgrag.utils.load_labels import get_full_dict
from tgrag.utils.matching import lookup_exact


def keep_unique(
    in_path: str, out_path: str, sort_cmd: str, tmpdir: str, mem: str
) -> None:
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    cmd = [sort_cmd, '-S', mem, '-T', tmpdir, '-u', in_path]
    with open(out_path, 'w', encoding='utf-8', newline='') as fout:
        p = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.PIPE, text=True, env=env
        )
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"sort failed: {' '.join(cmd)}\n{err}")


def external_sort_by_col(
    in_path: str,
    out_path: str,
    key_start_col: int,
    numeric: bool,
    sort_cmd: str,
    tmpdir: str,
    mem: str,
) -> None:
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    key = f'{key_start_col},{key_start_col}' + ('n' if numeric else '')
    cmd = [sort_cmd, '-S', mem, '-T', tmpdir, '-t', '\t', '-k', key, in_path]
    with open(out_path, 'w', encoding='utf-8', newline='') as fout:
        p = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.PIPE, text=True, env=env
        )
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"sort failed: {' '.join(cmd)}\n{err}")


def add_IDs_to_edges(edges_gz: str, src_idx_path: str, dst_idx_path: str) -> int:
    E = 0
    with (
        open(src_idx_path, 'w', encoding='utf-8', newline='') as fs,
        open(dst_idx_path, 'w', encoding='utf-8', newline='') as fd,
    ):
        for i, line in enumerate(gz_line_reader(edges_gz)):
            try:
                src, dst = line.split('\t', 1)
            except ValueError:
                continue
            fs.write(f'{i}\t{src.strip()}\n')
            fd.write(f'{i}\t{dst.strip()}\n')
            E = i + 1
    return E


def assign_IDs(
    domains_sorted: str, out_with_ids: str, e_gz: str, src_idx: str, dst_idx: str
) -> Tuple[int, int]:
    V = 0
    with (
        open(domains_sorted, 'r', encoding='utf-8', newline='') as fin,
        open(out_with_ids, 'w', encoding='utf-8', newline='') as fout,
    ):
        for i, line in enumerate(fin):
            dom = line.rstrip('\n')
            fout.write(f'{dom}\t{i}\n')
            V = i + 1

    E = add_IDs_to_edges(e_gz, src_idx, dst_idx)  # IDs in edges
    return V, E


def merge_join_index_to_IDs(
    sorted_idx_dom: str, sorted_dom_ids: str, out_idx_id: str
) -> None:
    with (
        open(sorted_idx_dom, 'r', encoding='utf-8', newline='') as fleft,
        open(sorted_dom_ids, 'r', encoding='utf-8', newline='') as fright,
        open(out_idx_id, 'w', encoding='utf-8', newline='') as fout,
    ):
        r_line = fright.readline()
        if not r_line:
            raise RuntimeError('domains.with_ids is empty')
        r_dom, r_id = r_line.rstrip('\n').split('\t')
        for l in fleft:
            l_idx, l_dom = l.rstrip('\n').split('\t')
            while r_dom < l_dom:
                r_line = fright.readline()
                if not r_line:
                    raise RuntimeError(f"domain '{l_dom}' not found in domains list")
                r_dom, r_id = r_line.rstrip('\n').split('\t')
            if r_dom != l_dom:
                raise RuntimeError(f"domain '{l_dom}' not found (mismatch).")
            fout.write(f'{l_idx}\t{r_id}\n')


def combine_IDs_by_index(
    src_idx_id_sorted: str, dst_idx_id_sorted: str, out_edges_ids: str
) -> int:
    E = 0
    with (
        open(src_idx_id_sorted, 'r', encoding='utf-8', newline='') as fs,
        open(dst_idx_id_sorted, 'r', encoding='utf-8', newline='') as fd,
        open(out_edges_ids, 'w', encoding='utf-8', newline='') as fout,
    ):
        s = fs.readline()
        d = fd.readline()
        while s and d:
            s_idx, s_id = s.rstrip('\n').split('\t')
            d_idx, d_id = d.rstrip('\n').split('\t')
            si = int(s_idx)
            di = int(d_idx)
            if si == di:
                fout.write(f'{s_id}\t{d_id}\n')
                E += 1
                s = fs.readline()
                d = fd.readline()
            elif si < di:
                s = fs.readline()
            else:
                d = fd.readline()
    return E


def print_pre_stats(V: int, E: int, deg: np.memmap, self_loops: int) -> None:
    iso = int((deg == 0).sum())
    leaves = int((deg == 1).sum())
    mean_deg = (2.0 * E / V) if V else 0.0
    max_deg = int(deg.max()) if V else 0
    min_deg = int(deg.min()) if V else 0
    undir_sparsity = (E / (V * (V - 1) / 2.0)) if V > 1 else 0.0
    print('[INFO] Pre-filter:')
    print(f'       V0={V}, E0={E}, iso={iso}, leaves={leaves}')
    print(f'       degree(min/mean/max)={min_deg}/{mean_deg:.4g}/{max_deg}')
    print(f'       sparsity={undir_sparsity:.6g}')
    print(f'       self-loops={self_loops}')


def print_post_stats(
    V0: int, E0: int, kept_V: int, kept_E: int, kept_self: int, kept_deg_vec: np.ndarray
) -> None:
    kept_V_pct = (100.0 * kept_V / V0) if V0 else 0.0
    kept_E_pct = (100.0 * kept_E / E0) if E0 else 0.0
    undir_sparsity = (kept_E / (kept_V * (kept_V - 1) / 2.0)) if kept_V > 1 else 0.0
    if kept_V > 0:
        min_k = int(kept_deg_vec.min())
        max_k = int(kept_deg_vec.max())
        mean_k = 2.0 * kept_E / kept_V
        iso_k = int((kept_deg_vec == 0).sum())
        leaf_k = int((kept_deg_vec == 1).sum())
    else:
        min_k = max_k = iso_k = leaf_k = 0
        mean_k = 0.0

    print('[INFO] After filtering:')
    print(
        f'       V={kept_V} ({kept_V_pct:.2f}% kept), E={kept_E} ({kept_E_pct:.2f}% kept)'
    )
    print(f'       sparsity={undir_sparsity:.6g}')
    print(f'       kept self-loops={kept_self}')
    print(f'       degree(min/mean/max)={min_k}/{mean_k:.4g}/{max_k}')
    print(f'       kept iso={iso_k}, kept leaves={leaf_k}')


def discard_deg(
    domains_with_ids: str, edges_ids_path: str, out_dir: str, k: int, slice_str: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    V = count_lines(domains_with_ids)

    # STATS (PRE)
    deg_path = os.path.join(out_dir, 'degree.int64.bin')
    deg = np.memmap(deg_path, dtype=np.int64, mode='w+', shape=(V,))
    deg[:] = 0

    E0 = 0
    self_loops_all = 0
    with open(edges_ids_path, 'r', encoding='utf-8', newline='') as fin:
        for line in fin:
            E0 += 1
            s, t = line.rstrip('\n').split('\t')
            si = int(s)
            ti = int(t)
            if si == ti:
                deg[si] += 2
                self_loops_all += 1
            else:
                deg[si] += 1
                deg[ti] += 1

    print_pre_stats(V, E0, deg, self_loops_all)

    # filter to deg > k
    edges_out_gz = os.path.join(out_dir, 'edges.txt.gz')
    kept_E = kept_self = 0
    with (
        gzip.open(edges_out_gz, 'wt', encoding='utf-8', newline='') as fout,
        open(edges_ids_path, 'r', encoding='utf-8', newline='') as fin,
    ):
        for line in fin:
            s, t = line.rstrip('\n').split('\t')
            si = int(s)
            ti = int(t)
            if deg[si] > k and deg[ti] > k:
                kept_E += 1
                if si == ti:
                    kept_self += 1
                fout.write(f'{si}\t{ti}\n')

    verts_out_gz = os.path.join(out_dir, 'vertices.txt.gz')
    kept_V = 0
    kept_deg = []
    with (
        gzip.open(verts_out_gz, 'wt', encoding='utf-8', newline='') as fout,
        open(domains_with_ids, 'r', encoding='utf-8', newline='') as fin,
    ):
        for line in fin:
            dom, sid = line.rstrip('\n').split('\t')
            si = int(sid)
            if deg[si] > k:
                kept_V += 1
                kept_deg.append(deg[si])
                fout.write(f'{si}\t{dom}\n')

    kept_deg_vec = (
        np.array(kept_deg, dtype=np.int64)
        if kept_deg
        else np.zeros((0,), dtype=np.int64)
    )
    print_post_stats(V, E0, kept_V, kept_E, kept_self, kept_deg_vec)

    # YYYYMMDD timestamp
    ts = iso_week_to_timestamp(slice_str)
    v_csv = os.path.join(out_dir, 'vertices.csv.gz')
    e_csv = os.path.join(out_dir, 'edges.csv.gz')
    with (
        gzip.open(verts_out_gz, 'rt', encoding='utf-8', newline='') as fin,
        gzip.open(v_csv, 'wt', encoding='utf-8', newline='') as fout,
    ):
        fout.write('nid,domain,ts\n')
        for line in fin:
            nid, dom = line.rstrip('\n').split('\t')
            fout.write(f'{nid},{dom},{ts}\n')

    with (
        gzip.open(edges_out_gz, 'rt', encoding='utf-8', newline='') as fin,
        gzip.open(e_csv, 'wt', encoding='utf-8', newline='') as fout,
    ):
        fout.write('src,dst,ts\n')
        for line in fin:
            s, t = line.rstrip('\n').split('\t')
            fout.write(f'{s},{t},{ts}\n')

    print(f'[INFO] Wrote CSVs with timestamps to {v_csv} and {e_csv}')


def generate_targets(output_path: str) -> None:
    vertices = os.path.join(output_path, 'vertices.csv.gz')
    targets_path = os.path.join(output_path, 'targets.csv')
    dqr_domains = get_full_dict()
    targets: Dict[int, List[float]] = {}

    with gzip.open(vertices, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split(',')
            # result = lookup(parts[1].strip(), dqr_domains)
            result = lookup_exact(parts[1].strip(), dqr_domains)
            if result is not None:
                targets[int(parts[0].strip())] = result

    with open(targets_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                'nid',
                'pc1',
                'afm',
                'afm_bias',
                'afm_min',
                'afm_rely',
                'fc',
                'mbfc',
                'mbfc_bias',
                'mbfc_fact',
                'mbfc_min',
                'lewandowsky_acc',
                'lewandowsky_trans',
                'lewandowsky_rely',
                'lewandowsky_mean',
                'lewandowsky_min',
                'misinfome_bin',
            ]
        )
        for nid, values in targets.items():
            writer.writerow([nid, *values])

    print(f'[INFO] Wrote targets to {targets_path}')


def process_graph(
    graph_path: str,
    min_deg: int,
    slice_str: str,
    only_targets: bool = False,
    sort_cmd: str = 'sort',
    mem: str = '60%',
) -> None:
    v_gz = os.path.join(graph_path, 'vertices.txt.gz')
    e_gz = os.path.join(graph_path, 'edges.txt.gz')
    if not (os.path.exists(v_gz) and os.path.exists(e_gz)):
        print(
            'expected vertices.txt.gz and edges.txt.gz under --graph-path',
            file=sys.stderr,
        )
        sys.exit(2)

    out_dir = os.path.join(graph_path, f'processed-deg{min_deg}')
    os.makedirs(out_dir, exist_ok=True)

    if not only_targets:
        with tempfile.TemporaryDirectory(prefix='extsort_') as td:
            all_domains = os.path.join(td, 'all_domains.txt')
            domains_sorted = os.path.join(td, 'domains.sorted.txt')
            domains_ids = os.path.join(td, 'domains.with_ids.txt')
            src_idx = os.path.join(td, 'src_idx.tsv')
            dst_idx = os.path.join(td, 'dst_idx.tsv')
            src_idx_sorted = os.path.join(td, 'src_idx.sorted.tsv')
            dst_idx_sorted = os.path.join(td, 'dst_idx.sorted.tsv')
            src_idx_id = os.path.join(td, 'src_idx_id.tsv')
            dst_idx_id = os.path.join(td, 'dst_idx_id.tsv')
            src_idx_id_sorted = os.path.join(td, 'src_idx_id.sorted.tsv')
            dst_idx_id_sorted = os.path.join(td, 'dst_idx_id.sorted.tsv')
            edges_ids = os.path.join(td, 'edges_ids.tsv')

            print('[STEP] extracting & keep unique set of domains')
            extract_all_domains(v_gz, e_gz, all_domains)
            keep_unique(all_domains, domains_sorted, sort_cmd, td, mem=mem)

            print('[STEP] assign numeric IDs')
            V, E = assign_IDs(domains_sorted, domains_ids, e_gz, src_idx, dst_idx)
            print(f'[INFO] unique domains (V) = {V}')
            print(f'[INFO] E ~ {E}')

            # sorting src_idx, dst_idx by domain to merge-join with domains.with_ids
            external_sort_by_col(
                src_idx,
                src_idx_sorted,
                key_start_col=2,
                numeric=False,
                sort_cmd=sort_cmd,
                tmpdir=td,
                mem=mem,
            )
            external_sort_by_col(
                dst_idx,
                dst_idx_sorted,
                key_start_col=2,
                numeric=False,
                sort_cmd=sort_cmd,
                tmpdir=td,
                mem=mem,
            )

            # joining to replace domains with IDs
            merge_join_index_to_IDs(src_idx_sorted, domains_ids, src_idx_id)
            merge_join_index_to_IDs(dst_idx_sorted, domains_ids, dst_idx_id)

            # sort back by edge index to zip
            external_sort_by_col(
                src_idx_id,
                src_idx_id_sorted,
                key_start_col=1,
                numeric=True,
                sort_cmd=sort_cmd,
                tmpdir=td,
                mem=mem,
            )
            external_sort_by_col(
                dst_idx_id,
                dst_idx_id_sorted,
                key_start_col=1,
                numeric=True,
                sort_cmd=sort_cmd,
                tmpdir=td,
                mem=mem,
            )

            # combine src/dst ids to final edges
            combine_IDs_by_index(src_idx_id_sorted, dst_idx_id_sorted, edges_ids)

            print('[STEP] discard deg<k & write outputs')
            discard_deg(domains_ids, edges_ids, out_dir, min_deg, slice_str)

    # try:
    #     print('[STEP] generating targets.csv')
    #     generate_targets(out_dir)

    # except Exception as e:
    #     print(
    #         f'[ERROR] failed to generate targets.csv; make sure vertices.csv.gz exists in the output directory: {e}'
    #     )

    print('[DONE] outputs in:', out_dir)
