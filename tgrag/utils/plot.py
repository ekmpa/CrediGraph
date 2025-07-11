import logging
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from scipy.ndimage import gaussian_filter

from tgrag.utils.path import get_root_dir


def plot_avg_rmse_loss(
    loss_tuple_run: List[List[Tuple[float, float, float]]],
    model_name: str,
    save_filename: str = 'rmse_loss_plot.png',
) -> None:
    """Plots the averaged RMSE loss over trials for train, validation, and test sets.

    Parameters:
    - loss_tuple_run: List of runs (trials), each a list of (train, val, test) RMSE tuples per epoch.
    - model_name: The name of the model
    - save_filename: Name of the generated plot (name of png file).
    """
    num_epochs = len(loss_tuple_run[0])

    data = np.array(loss_tuple_run)  # shape: (num_trials, num_epochs, 3)

    avg_rmse = data.mean(axis=0)  # shape: (num_epochs, 3)

    avg_train = avg_rmse[:, 0]
    avg_val = avg_rmse[:, 1]
    avg_test = avg_rmse[:, 2]

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    plt.plot(epochs, avg_train, label='Train RMSE', linewidth=2)
    plt.plot(epochs, avg_val, label='Validation RMSE', linewidth=2)
    plt.plot(epochs, avg_test, label='Test RMSE', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title(f'{model_name} : Average RMSE Loss over Trials')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_all_loss_tuples(
    results_dir: str = 'results/logs',
) -> Dict[str, List[List[Tuple[float, float, float]]]]:
    """Loads all loss_tuple_run.pkl files from results/logs/MODEL/ENCODER.

    Returns:
        A dictionary mapping "MODEL_ENCODER" to the corresponding loss data.
    """
    results = {}
    root = get_root_dir()
    base_path = root / results_dir
    for model_dir in base_path.iterdir():
        for encoder_dir in model_dir.iterdir():
            file_path = encoder_dir / 'loss_tuple_run.pkl'
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    loss_data = pickle.load(f)
                key = f'{model_dir.name}_{encoder_dir.name}'
                results[key] = loss_data
    return results


def plot_metric_across_models(
    all_results: Dict[str, List[List[Tuple[float, float, float]]]],
    metric: str = 'test',  # "train", "valid", or "test"
    save_filename: str = 'compare_models.png',
) -> None:
    """Plots the selected metric across models over epochs.

    Args:
        all_results: Dict from model_encoder to loss_tuple_run.
        metric: One of "train", "valid", or "test".
        save_filename: Name of the saved file (name of the png).
    """
    metric_index = {'train': 0, 'valid': 1, 'test': 2}[metric]

    plt.figure(figsize=(10, 6))

    for label, loss_tuple_run in all_results.items():
        data = np.array(loss_tuple_run)  # shape: (runs, epochs, 3)
        avg_over_runs = data.mean(axis=0)  # shape: (epochs, 3)
        metric_values = avg_over_runs[:, metric_index]
        epochs = np.arange(1, len(metric_values) + 1)
        plt.plot(epochs, metric_values, label=label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.capitalize()} RMSE')
    plt.title(f'Comparison of {metric.capitalize()} RMSE Across Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    root = get_root_dir()
    save_dir = root / 'results' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.savefig(save_path)
    plt.close()


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
