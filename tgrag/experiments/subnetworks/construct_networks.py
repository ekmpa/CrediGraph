#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

from tgrag.utils.data_loading import shlex_quote


def sort_file_inplace_tsv(file_path: str, key_fields: str = '-k1,1') -> None:
    """Enforce LC_ALL=C byte-order sorting on a TSV file."""
    if not os.path.exists(file_path):
        return

    tmp = file_path + '.sorted'
    env = os.environ.copy()
    env['LC_ALL'] = 'C'

    key_parts = key_fields.split()

    cmd = ['sort', '-t', '\t'] + key_parts + [file_path]
    with open(tmp, 'wb') as out_f:
        subprocess.check_call(cmd, env=env, stdout=out_f)

    os.replace(tmp, file_path)


def assert_sorted_tsv(file_path: str, key_fields: str = '-k1,1') -> None:
    """Validate that a TSV file is sorted lexicographically by the given keys."""
    if not os.path.exists(file_path):
        return

    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    key_parts = key_fields.split()

    cmd = ['sort', '-c', '-t', '\t'] + key_parts + [file_path]
    subprocess.check_call(cmd, env=env)


def run_checked(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print(f"[DEBUG] Running: {' '.join(shlex_quote(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _atomic_write_from_pipeline(
    cmds: List[List[str]], out_path: str, env: Optional[Dict[str, str]] = None
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix='.tmp_', suffix='.gz', dir=os.path.dirname(out_path), delete=False
    ) as tmp:
        tmp_path = tmp.name

    procs = []
    prev_stdout = None
    try:
        for i, argv in enumerate(cmds):
            print(
                f"[DEBUG] Pipeline stage {i}: {' '.join(shlex_quote(c) for c in argv)}"
            )
            p = subprocess.Popen(
                argv,
                stdin=prev_stdout,
                stdout=(subprocess.PIPE if i < len(cmds) - 1 else open(tmp_path, 'wb')),
                env=env,
            )
            if prev_stdout is not None:
                prev_stdout.close()
            prev_stdout = p.stdout
            procs.append(p)

        retcode = 0
        for p in procs[::-1]:
            p.wait()
            if p.returncode != 0:
                retcode = p.returncode
        if retcode != 0:
            raise subprocess.CalledProcessError(
                retcode, ' | '.join(' '.join(c) for c in cmds)
            )

        os.replace(tmp_path, out_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def preprocess_edges(
    edges_csv_gz: str, out_edges_tsv_gz: str, env: Dict[str, str]
) -> None:
    print(f'[INFO] Preprocessing edges → {out_edges_tsv_gz}')
    cmds = [
        ['gzip', '-cd', '--', edges_csv_gz],
        ['awk', '-F,', 'NR>1 {print $1 "\t" $2}'],
        ['sort', '-t', '\t', '-k1,1'],
        ['gzip', '-c'],
    ]
    _atomic_write_from_pipeline(cmds, out_edges_tsv_gz, env=env)


def make_frontier0(seeds_tsv: str, frontier0_tsv: str, env: Dict[str, str]) -> None:
    print(f'[INFO] Building depth-0 frontier → {frontier0_tsv}')
    bash_script = f"""
set -euo pipefail
awk -F $'\\t' '{{print $1"\\t"$1"\\t0\\t"$2}}' {shlex_quote(seeds_tsv)} |
    sort -t $'\\t' -k1,1 -k2,2 > {shlex_quote(frontier0_tsv)}
"""
    run_checked(['/bin/bash', '-lc', bash_script], env)

    # verify + enforce sort
    sort_file_inplace_tsv(frontier0_tsv, '-k1,1 -k2,2')
    assert_sorted_tsv(frontier0_tsv, '-k1,1 -k2,2')


def bfs_step(
    depth: int,
    edges_by_src_tsv_gz: str,
    frontier_tsv: str,
    next_frontier_tsv: str,
    out_edges_tsv_gz: str,
    env: Dict[str, str],
) -> None:
    print(f'[INFO] BFS depth {depth} → {depth+1}')

    # frontier must be sorted by node
    assert_sorted_tsv(frontier_tsv, '-k1,1')

    bash_script = f"""
set -euo pipefail

gzip -cd -- {shlex_quote(edges_by_src_tsv_gz)} |
  join -t $'\\t' -1 1 -2 1 - {shlex_quote(frontier_tsv)} |
  tee >(
      awk -F $'\\t' '{{ printf "%s\\t%s\\t%d\\t%s\\n", $2, $3, $4+1, $5 }}' |
      sort -t $'\\t' -k1,1 -k2,2 -u > {shlex_quote(next_frontier_tsv)}
  ) |
  awk -F $'\\t' '{{ printf "%s\\t%s\\t%s\\t%d\\t%d\\t%s\\n", $3, $1, $2, $4, $4+1, $5 }}' |
  gzip -c
"""

    _atomic_write_from_pipeline(
        [['/bin/bash', '-lc', bash_script]], out_edges_tsv_gz, env
    )

    # enforce sorting on next frontier
    sort_file_inplace_tsv(next_frontier_tsv, '-k1,1 -k2,2')
    assert_sorted_tsv(next_frontier_tsv, '-k1,1 -k2,2')


def merge_edges(
    edge_files_gz: List[str], out_merged_gz: str, env: Dict[str, str]
) -> None:
    print(f'[INFO] Merging edges → {out_merged_gz}')
    cmds = [
        ['gzip', '-cd', '--', *edge_files_gz],
        ['sort', '-t', '\t', '-k1,1', '-k2,2', '-k3,3', '-k4,4n', '-k5,5n'],
        ['gzip', '-c'],
    ]
    _atomic_write_from_pipeline(cmds, out_merged_gz, env)


def merge_nodes(
    frontier_files: List[str], out_nodes_tsv: str, env: Dict[str, str]
) -> None:
    """Merge frontier_* files (node,root,depth,pc1) into:
      node<TAB>root<TAB>min_depth<TAB>pc1_root.

    With streaming and external sort.
    """
    print(f'[INFO] Merging node frontiers → {out_nodes_tsv}')
    tmp = out_nodes_tsv + '.tmp'

    awk_first_per_key = r"""
        BEGIN { FS="\t"; OFS="\t"; prev="" }
        {
          key = $2 FS $1;      # root \t node
          if (key != prev) {
            print $1, $2, $3, $4;  # node, root, depth(min), pc1
            prev = key;
          }
        }
    """

    bash_script = f"""
set -euo pipefail

# Sort by (root, node, depth) so first row per (root,node) has min depth
cat {" ".join(shlex_quote(f) for f in frontier_files)} |
  LC_ALL=C sort -T {shlex_quote(os.path.dirname(out_nodes_tsv))} -S 50% -t $'\\t' -k2,2 -k1,1 -k3,3n |
  awk '{awk_first_per_key}' > {shlex_quote(tmp)}
"""
    run_checked(['/bin/bash', '-lc', bash_script], env)

    os.replace(tmp, out_nodes_tsv)

    sort_file_inplace_tsv(out_nodes_tsv, '-k2,2 -k1,1')
    assert_sorted_tsv(out_nodes_tsv, '-k2,2 -k1,1')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', required=True)
    parser.add_argument('--ratings', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    scratch = os.environ.get('SCRATCH', None)
    if scratch is None:
        raise RuntimeError('SCRATCH environment variable not set.')

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    print(f'[INFO] Output directory: {outdir}')

    env = os.environ.copy()
    env['LC_ALL'] = 'C'

    edges_by_src = os.path.join(outdir, 'edges_by_src.tsv.gz')
    seeds_tsv = os.path.join(outdir, 'seeds_graph_native.tsv')
    frontier0 = os.path.join(outdir, 'frontier_depth0.tsv')
    frontier1 = os.path.join(outdir, 'frontier_depth1.tsv')
    frontier2 = os.path.join(outdir, 'frontier_depth2.tsv')
    edges_d0 = os.path.join(outdir, 'edges_depth0.tsv.gz')
    edges_d1 = os.path.join(outdir, 'edges_depth1.tsv.gz')
    merged_edges = os.path.join(outdir, 'subgraph_edges.tsv.gz')
    merged_nodes = os.path.join(outdir, 'subgraph_nodes.tsv')

    if not os.path.exists(edges_by_src):
        preprocess_edges(args.edges, edges_by_src, env)
    else:
        print('[INFO] Reusing edges_by_src')

    if not os.path.exists(seeds_tsv):
        raise RuntimeError(
            f'Missing {seeds_tsv} — must run build_graph_native_seeds.py first.'
        )

    print(f'[INFO] Using seeds from {seeds_tsv}')
    sort_file_inplace_tsv(seeds_tsv, '-k1,1')
    assert_sorted_tsv(seeds_tsv, '-k1,1')

    # Build frontier 0
    if not os.path.exists(frontier0):
        make_frontier0(seeds_tsv, frontier0, env)
    else:
        print('[INFO] Reusing frontier0')
        assert_sorted_tsv(frontier0, '-k1,1 -k2,2')

    # BFS levels
    bfs_step(0, edges_by_src, frontier0, frontier1, edges_d0, env)
    bfs_step(1, edges_by_src, frontier1, frontier2, edges_d1, env)

    # Merge results
    merge_edges([edges_d0, edges_d1], merged_edges, env)
    merge_nodes([frontier0, frontier1, frontier2], merged_nodes, env)

    print('[INFO] Finished.')
    print(f'[INFO] Edges → {merged_edges}')
    print(f'[INFO] Nodes  → {merged_nodes}')


if __name__ == '__main__':
    main()
