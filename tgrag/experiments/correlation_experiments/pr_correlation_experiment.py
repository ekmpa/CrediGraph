import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from tgrag.utils.args import DataArguments
from tgrag.utils.path import get_root_dir


def run_pr_cr_bin_correlation(data_arguments: DataArguments) -> None:
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
        return

    df = pd.DataFrame(collected_data, columns=['pr_value', 'cr_score'])

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pr_labels = ['v_low_pr', 'low_pr', 'mid_pr', 'high_pr', 'v_high_pr']
    cr_labels = ['v_low_cr', 'low_cr', 'mid_cr', 'high_cr', 'v_high_cr']

    df['pr_bin'] = pd.cut(
        df['pr_value'], bins=bins, labels=pr_labels, include_lowest=True, right=False
    )
    df['cr_bin'] = pd.cut(
        df['cr_score'], bins=bins, labels=cr_labels, include_lowest=True, right=False
    )

    contingency = pd.crosstab(df['cr_bin'], df['pr_bin'], normalize='all')

    root = get_root_dir()
    save_dir = root / 'results' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_bin_correlation_heatmap.png'

    sns.heatmap(contingency, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('PR/CR Binned Correlation Heatmap')
    plt.xlabel('PageRank Bins')
    plt.ylabel('Credibility Score Bins')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f'Binned heatmap saved to: {save_path}')


def run_pr_correlation(data_arguments: DataArguments) -> None:
    node_file = data_arguments.node_file
    chunk_size = 100_000  # Tune based on memory
    collected_data = []

    logging.info(
        'Setting up training for task of: %s',
        data_arguments.task_name,
    )

    for chunk in tqdm(
        pd.read_csv(node_file, chunksize=chunk_size), desc='Processing chunks'
    ):
        filtered = chunk[(chunk['cr_score'] != -1)]

        collected_data.extend(filtered[['pr_value', 'cr_score']].values.tolist())

        if not filtered.empty:
            collected_data.extend(filtered[['pr_value', 'cr_score']].values.tolist())

    if not collected_data:
        logging.info('No valid rows with both pr_value and cr_score found.')
        return

    df = pd.DataFrame(collected_data, columns=['pr_value', 'cr_score'])

    correlation_matrix = df.corr(method='pearson')

    root = get_root_dir()
    save_dir = root / 'results' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_correlation_heatmap.png'
    sns.heatmap(correlation_matrix, annot=True, fmt='.5f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap: pr_value vs cr_score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f'Heatmap saved to: {save_path}')
