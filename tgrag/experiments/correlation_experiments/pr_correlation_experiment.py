import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from tgrag.utils.args import DataArguments
from tgrag.utils.path import get_root_dir


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


def plot_pr_cr_bin_correlation(heat_map: DataFrame) -> None:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pr_labels = ['v_low_pr', 'low_pr', 'mid_pr', 'high_pr', 'v_high_pr']
    cr_labels = ['v_low_cr', 'low_cr', 'mid_cr', 'high_cr', 'v_high_cr']

    heat_map['pr_bin'] = pd.cut(
        heat_map['pr_value'],
        bins=bins,
        labels=pr_labels,
        include_lowest=True,
        right=False,
    )
    heat_map['cr_bin'] = pd.cut(
        heat_map['cr_score'],
        bins=bins,
        labels=cr_labels,
        include_lowest=True,
        right=False,
    )

    contingency = pd.crosstab(heat_map['cr_bin'], heat_map['pr_bin'], normalize='all')

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_bin_correlation_heatmap.png'

    sns.heatmap(contingency, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('PR/CR Binned Correlation Heatmap')
    plt.xlabel('PageRank Bins')
    plt.ylabel('Credibility Score Bins')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f'Binned heat_map saved to: {save_path}')


def plot_pr_correlation(heat_map: DataFrame) -> None:
    numeric_cols = heat_map.select_dtypes(include=[np.number])
    correlation_matrix = numeric_cols.corr(method='pearson')

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_correlation_heatmap.png'
    sns.heatmap(correlation_matrix, annot=True, fmt='.5f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap: pr_value vs cr_score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f'Heatmap saved to: {save_path}')


def plot_pr_correlation_log_scale(heat_map: DataFrame) -> None:
    epsilon = 1e-10  # To avoid log(0)
    heat_map['log_pr'] = np.log(heat_map['pr_value'] + epsilon)

    pr_labels = ['v_low_pr', 'low_pr', 'mid_pr', 'high_pr', 'v_high_pr']
    cr_labels = ['v_low_cr', 'low_cr', 'mid_cr', 'high_cr', 'v_high_cr']

    heat_map['pr_bin'] = pd.qcut(heat_map['log_pr'], q=5, labels=pr_labels)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    heat_map['cr_bin'] = pd.cut(
        heat_map['cr_score'],
        bins=bins,
        labels=cr_labels,
        include_lowest=True,
        right=False,
    )

    contingency = pd.crosstab(heat_map['cr_bin'], heat_map['pr_bin'], normalize='all')

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_bin_correlation_heatmap_log_scale.png'

    sns.heatmap(contingency, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('PR/CR Binned Correlation Heatmap (Log PageRank Bins)')
    plt.xlabel('PageRank Bins (log-scale quantiles)')
    plt.ylabel('Credibility Score Bins')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f'Binned heatmap saved to: {save_path}')


def plot_pr_correlation_auto_bin(heat_map: DataFrame) -> None:
    pr_labels = ['v_low_pr', 'low_pr', 'mid_pr', 'high_pr', 'v_high_pr']
    cr_labels = ['v_low_cr', 'low_cr', 'mid_cr', 'high_cr', 'v_high_cr']

    min_pr, max_pr = heat_map['pr_value'].min(), heat_map['pr_value'].max()
    if max_pr - min_pr < 1e-6:
        logging.warning('Not enough variation in pr_value to create bins.')
        return
    pr_bins = np.linspace(min_pr, max_pr, num=6)  # 5 bins = 6 edges
    heat_map['pr_bin'] = pd.cut(
        heat_map['pr_value'],
        bins=pr_bins,
        labels=pr_labels,
        include_lowest=True,
        right=False,
    )

    min_cr, max_cr = heat_map['cr_score'].min(), heat_map['cr_score'].max()
    if max_cr - min_cr < 1e-6:
        logging.warning('Not enough variation in cr_score to create bins.')
        return
    cr_bins = np.linspace(min_cr, max_cr, num=6)
    heat_map['cr_bin'] = pd.cut(
        heat_map['cr_score'],
        bins=cr_bins,
        labels=cr_labels,
        include_lowest=True,
        right=False,
    )

    contingency = pd.crosstab(heat_map['cr_bin'], heat_map['pr_bin'], normalize='all')

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_bin_correlation_heatmap_adaptive_both.png'

    sns.heatmap(contingency, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('PR/CR Binned Correlation Heatmap (Adaptive PR & CR Bins)')
    plt.xlabel('PageRank Bins (adaptive)')
    plt.ylabel('Credibility Score Bins (adaptive)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f'Adaptive PR/CR binning heatmap saved to: {save_path}')


def plot_joint_pr_cr_heatmap(df: pd.DataFrame) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    df['pr_bin'] = (100 * df['pr_value']).round(1).astype(int)
    df['cr_bin'] = (100 * df['cr_score']).round(1).astype(int)

    df = df[(df['pr_bin'] >= 0) & (df['pr_bin'] <= 100)]
    df = df[(df['cr_bin'] >= 0) & (df['cr_bin'] <= 100)]

    joint_counts = df.groupby(['cr_bin', 'pr_bin']).size().reset_index(name='count')

    heat = np.zeros((10, 10))  # 10x10 bins
    for i in range(10):
        for j in range(10):
            cr_val = i * 10
            pr_val = j * 10
            cell = joint_counts[
                (joint_counts['cr_bin'] == cr_val) & (joint_counts['pr_bin'] == pr_val)
            ]['count'].values
            heat[i, j] = cell[0] if len(cell) else 0

    # Apply Gaussian smoothing for interpretability
    smoothed = gaussian_filter(heat, sigma=1.5)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        smoothed,
        cmap='coolwarm_r',
        vmin=smoothed.min(),
        vmax=smoothed.max(),
        cbar_kws={'label': 'Node Count (smoothed)'},
    )

    ax.set_title('Joint PR–CR Distribution')
    ax.set_xlabel('PageRank Group')
    ax.set_ylabel('Credibility Score Group')

    ax.set_xticks([0, 5, 9])
    ax.set_xticklabels([0.0, 0.5, 1.0])

    ax.set_yticks([0, 5, 9])
    ax.set_yticklabels([1.0, 0.5, 0.0])  # Flip to match matrix orientation

    plt.tight_layout()
    plt.savefig(save_dir / 'joint_pr_cr_heatmap.png', dpi=300)
    plt.close()


def run_correlation(data_arguments: DataArguments) -> None:
    heat_map = get_heat_matrix(data_arguments)
    plot_pr_cr_bin_correlation(heat_map)
    plot_pr_correlation_log_scale(heat_map)
    plot_pr_correlation(heat_map)
    plot_pr_correlation_auto_bin(heat_map)
    plot_joint_pr_cr_heatmap(heat_map)
