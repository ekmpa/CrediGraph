import logging

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from tgrag.utils.args import DataArguments
from tgrag.utils.plot import (
    plot_joint_pr_cr_heatmap,
    plot_pr_correlation,
    plot_pr_correlation_auto_bin,
    plot_pr_correlation_log_scale,
    plot_pr_cr_bin_correlation,
)


def get_heat_matrix(data_arguments: DataArguments) -> DataFrame:
    node_file = data_arguments.node_file
    chunk_size = 100_000
    collected_data = []

    logging.info(
        'Setting up training for task of: %s',
        data_arguments.task_name,
    )

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


def run_correlation(data_arguments: DataArguments) -> None:
    heat_map = get_heat_matrix(data_arguments)
    plot_pr_cr_bin_correlation(heat_map)
    plot_pr_correlation_log_scale(heat_map)
    plot_pr_correlation(heat_map)
    plot_pr_correlation_auto_bin(heat_map)
    plot_joint_pr_cr_heatmap(heat_map)
