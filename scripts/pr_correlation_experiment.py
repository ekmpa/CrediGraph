import argparse
import logging

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from tgrag.utils.logger import setup_logging
from tgrag.utils.plot import (
    plot_joint_pr_cr_heatmap,
    plot_pr_correlation,
    plot_pr_correlation_auto_bin,
    plot_pr_correlation_log_scale,
    plot_pr_cr_bin_correlation,
    plot_pr_cr_scatter_logx,
    plot_pr_vs_cr_scatter,
    plot_spearman_pr_cr_correlation,
)

parser = argparse.ArgumentParser(
    description='PageRank Correlation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--node-file',
    type=str,
    default='data/crawl-data/manual/output/test_pr_cr.csv',
    help='Path to file containing raw node pagerank/credibility column data in CSV format',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='script_correlation.log',
    help='Name of log file at project root.',
)


def get_heat_matrix(node_file: str) -> DataFrame:
    chunk_size = 100_000
    collected_data = []

    logging.info('Streaming matrix construction.')

    for chunk in tqdm(
        pd.read_csv(node_file, chunksize=chunk_size), desc='Reading PD chunk'
    ):
        filtered = chunk[(chunk['cr_score'] >= 0)]
        if not filtered.empty:
            collected_data.extend(filtered[['pr_value', 'cr_score']].values.tolist())

    logging.info('All scores found.')

    if not collected_data:
        logging.info('No valid rows with pr_value and cr_score in [0,1).')

    return pd.DataFrame(collected_data, columns=['pr_value', 'cr_score'])


def run_correlation() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)
    heat_map = get_heat_matrix(args.node_file)
    plot_pr_cr_bin_correlation(heat_map)
    plot_pr_correlation_log_scale(heat_map)
    plot_pr_correlation(heat_map)
    plot_pr_correlation_auto_bin(heat_map)
    plot_joint_pr_cr_heatmap(heat_map)
    plot_spearman_pr_cr_correlation(heat_map)
    plot_pr_cr_scatter_logx(heat_map)
    plot_pr_vs_cr_scatter(heat_map)


if __name__ == '__main__':
    run_correlation()
