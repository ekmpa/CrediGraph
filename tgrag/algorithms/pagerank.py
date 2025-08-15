import argparse
import logging
from typing import Dict, List

import pandas as pd

from tgrag.utils.logger import setup_logging
from tgrag.utils.pagerank_utils import (
    build_adjacency,
    check_convergence,
    preprocess_data,
    show_score_distribution,
    test_degree_correlation,
    test_positive_values,
    test_score_sum,
)
from tgrag.utils.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Generate PageRank.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--node-file',
    type=str,
    help='Path to the feature file in a csv format.',
)
parser.add_argument(
    '--edge-file',
    type=str,
    help='Path to the edge file in a csv format.',
)
parser.add_argument(
    '--node-save-file',
    type=str,
    help='Path to the written feature file in a csv format.',
)
parser.add_argument(
    '--edge-save-file',
    type=str,
    help='Path to the written edge file in a csv format.',
)
parser.add_argument(
    '--damping',
    type=float,
    help='Damping paramater in Page Rank.',
)
parser.add_argument(
    '--max_iter',
    type=float,
    help='The maximum iterations.',
)
parser.add_argument(
    '--tolerance',
    type=float,
    help='The convergence tolerance.',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='script_pagerank.log',
    help='Name of log file at project root.',
)


def compute_new_importance(
    node_ids: List[int],
    prev_importance: pd.Series,
    out_degree: Dict[int, int],
    incoming: Dict[int, List[int]],
    damping: float,
    dangling_sum: float,
    N: int,
) -> pd.Series:
    new_importance = {}
    for node in node_ids:
        rank_sum = 0
        for src in incoming[node]:
            if src in prev_importance and out_degree.get(src, 0) > 0:
                rank_sum += prev_importance[src] / out_degree[src]
        dangling_contribution = dangling_sum / N

        # (1 - damping): probability of jumping to a random page
        new_importance[node] = (1 - damping) / N + damping * (
            rank_sum + dangling_contribution
        )
    return pd.Series(new_importance)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    setup_logging(args.log_file)
    nodes, edges = preprocess_data(root / args.node_file, root / args.edge_file)
    node_ids, adjacency, out_degree, incoming = build_adjacency(nodes, edges)
    logging.info('*** Running PageRank algorithm ***')
    N = len(node_ids)
    importance = pd.Series(1.0 / N, index=node_ids)
    iteration = 0
    converged = False
    while iteration < args.max_iter and not converged:
        prev_importance = importance.copy()
        dangling_sum = sum(
            prev_importance[node] for node in node_ids if out_degree.get(node, 0) == 0
        )
        importance = compute_new_importance(
            node_ids,
            prev_importance,
            out_degree,
            incoming,
            args.damping,
            dangling_sum,
            N,
        )
        converged = check_convergence(
            iteration, prev_importance, importance, args.tolerance
        )
        iteration += 1

    nodes['importance'] = nodes['node_id'].map(importance)

    convergence_test = converged
    score_sum = test_score_sum(nodes)
    positive_values_test = test_positive_values(nodes)
    degree_correlation_test = test_degree_correlation(nodes, edges)

    logging.info(f'Convergence test passed: {convergence_test}')
    logging.info(f'Score sum test passed: {score_sum}')
    logging.info(f'Positive values test passed: {positive_values_test}')
    logging.info(f'Degree correlation test passed: {degree_correlation_test}')

    show_score_distribution(nodes)

    nodes.to_csv(args.node_save_file, index=False)
    edges.to_csv(args.edge_save_file, index=False)
    logging.info(f'Results saved to {args.node_save_file} and {args.edge_save_file}')
