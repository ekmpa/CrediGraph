import argparse
import gzip
import os
from typing import Dict

# import preprocess_edges from your construct_networks.py
from construct_networks import preprocess_edges

from tgrag.utils.data_loading import load_dqr
from tgrag.utils.matching import flip_if_needed, reverse_domain


def ensure_edges_by_src(edges_csv_gz: str, edges_by_src_gz: str) -> None:
    """Create edges_by_src.tsv.gz (src<TAB>dst, sorted by src) if it does not exist."""
    if os.path.exists(edges_by_src_gz):
        print(f'[INFO] Reusing existing {edges_by_src_gz}')
        return

    print(
        f'[INFO] Building edges_by_src.tsv.gz from {edges_csv_gz} → {edges_by_src_gz}'
    )
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    preprocess_edges(edges_csv_gz, edges_by_src_gz, env)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build graph-native seeds from DQR.')
    ap.add_argument(
        '--edges_csv_gz',
        required=True,
        help='Path to edges.csv.gz (src,dst,ts with header).',
    )
    ap.add_argument(
        '--edges_by_src_gz',
        required=True,
        help='Path to edges_by_src.tsv.gz (src<TAB>dst, sorted by src). '
        'Will be created if it does not exist.',
    )
    ap.add_argument(
        '--dqr_csv',
        required=True,
        help="Path to domain_ratings.csv (must contain 'domain,pc1,...').",
    )
    ap.add_argument(
        '--out_tsv',
        required=True,
        help='Output seeds TSV: graph_node_id<TAB>pc1_root.',
    )
    args = ap.parse_args()

    ensure_edges_by_src(args.edges_csv_gz, args.edges_by_src_gz)

    print(f'[INFO] loading DQR ratings from {args.dqr_csv} ...')
    dqr = load_dqr(args.dqr_csv)

    print(f'[INFO] scanning graph nodes in {args.edges_by_src_gz} and matching...')
    matched: Dict[str, float] = {}
    total_seen = 0

    with gzip.open(args.edges_by_src_gz, 'rt') as f:
        for line in f:
            if not line:
                continue
            total_seen += 1
            src = line.split('\t', 1)[0].strip().lower()

            candidates = {  # TODO change this to just flip_if_needed.
                # just need to test.
                flip_if_needed(src),
                flip_if_needed(reverse_domain(src)),
                src,
            }
            for cand in candidates:
                pc1 = dqr.get(cand)
                if pc1 is not None:
                    matched[src] = pc1
                    break

    print(f'[INFO] scanned {total_seen} edges (src nodes)')
    print(f'[INFO] matched {len(matched)} graph-native roots')

    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    print(f'[INFO] writing seeds → {args.out_tsv}')
    with open(args.out_tsv, 'w') as out:
        for node, pc1 in sorted(matched.items()):
            out.write(f'{node}\t{pc1}\n')


if __name__ == '__main__':
    main()
