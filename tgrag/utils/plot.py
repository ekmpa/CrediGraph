import logging
import pickle
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas import DataFrame
from scipy.ndimage import gaussian_filter
from torch import Tensor

from tgrag.utils.path import get_root_dir


class Scoring(str, Enum):
    mse = 'MSE'
    r2 = 'R2'
    mae = 'MAE'


class Label(str, Enum):
    pc1 = 'PC1'
    mbfc = 'MBFC-BIAS'


def mean_across_lists(lists: list[list[float]]) -> list[float]:
    max_len = max(len(lst) for lst in lists)
    arr = np.full((len(lists), max_len), np.nan, dtype=float)
    for i, lst in enumerate(lists):
        arr[i, : len(lst)] = lst
    return np.nanmean(arr, axis=0).tolist()


def plot_pred_target_distributions_bin_list(
    preds: List[List[float]],
    targets: List[List[float]],
    model_name: str,
    title: str = 'True vs Predicted Distribution',
    save_filename: str = 'pred_target_distribution.png',
    bins: int = 50,
) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name / 'distribution'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    preds_flat = np.concatenate([np.array(p, dtype=float) for p in preds])
    targets_flat = np.concatenate([np.array(t, dtype=float) for t in targets])

    hist_true, _ = np.histogram(targets_flat, bins=bins, range=(0, 1))
    hist_pred, _ = np.histogram(preds_flat, bins=bins, range=(0, 1))
    y_max = max(hist_true.max(), hist_pred.max())

    plt.figure(figsize=(6, 4), dpi=120)
    plt.hist(
        preds_flat,
        bins=bins,
        range=(0, 1),
        edgecolor='black',
        color='lightblue',
        label='Pred',
        alpha=0.8,
    )
    plt.hist(
        targets_flat,
        bins=bins,
        range=(0, 1),
        edgecolor='black',
        color='orange',
        label='True',
        alpha=0.6,
    )

    plt.rc('font', size=13)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, y_max + 50, 100))
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_pred_target_distributions_bin(
    preds: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    title: str = 'Average distribution True vs Predicted',
    save_filename: str = 'pred_target_distribution.png',
    bins: int | np.ndarray = 40,
    xlim: tuple[float, float] = (0.0, 1.0),
    logy: bool = False,
) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name / 'distribution'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename
    p = preds.detach().flatten().to('cpu').float().numpy()
    t = targets.detach().flatten().to('cpu').float().numpy()

    # clean NaNs/Infs
    p = p[np.isfinite(p)]
    t = t[np.isfinite(t)]

    # fixed bins over the chosen x-range
    if isinstance(bins, int):
        bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    else:
        bin_edges = np.asarray(bins)

    plt.figure(figsize=(8, 6))
    plt.hist(t, bins=bin_edges, alpha=0.6, label='True', density=False)
    plt.hist(p, bins=bin_edges, alpha=0.6, label='Pred', density=False)

    if logy:
        plt.yscale('log')

    plt.xlim(*xlim)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_pred_target_distributions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    title: str = 'Average distribution True vs Predicted',
    save_filename: str = 'pred_target_distribution.png',
) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name / 'distribution'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    bins = 20  # adjust depending on granularity

    # plot normalized histograms (distributions)
    plt.hist(
        targets, bins=bins, alpha=0.6, label='True', color='tab:blue', density=True
    )
    plt.hist(
        preds, bins=bins, alpha=0.6, label='Pred', color='tab:orange', density=True
    )

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_avg_loss(
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]],
    model_name: str,
    score: Scoring,
    save_filename: str = 'loss_plot.png',
) -> None:
    """Plots the averaged MSE loss over trials for train, validation, and test sets with std dev bands."""
    num_epochs = len(loss_tuple_run[0])

    data = np.array(loss_tuple_run)  # shape: (num_trials, num_epochs, 3)

    avg = data.mean(axis=0)  # shape: (num_epochs, 3)
    std = data.std(axis=0)  # shape: (num_epochs, 3)

    avg_train, avg_val, avg_test, avg_mean, avg_random = (
        avg[:, 0],
        avg[:, 1],
        avg[:, 2],
        avg[:, 3],
        avg[:, 4],
    )
    std_train, std_val, std_test, std_mean, std_random = (
        std[:, 0],
        std[:, 1],
        std[:, 2],
        std[:, 3],
        std[:, 4],
    )

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    plt.plot(epochs, avg_train, label=f'Train {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_train - std_train, avg_train + std_train, alpha=0.2)

    plt.plot(epochs, avg_val, label=f'Validation {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_val - std_val, avg_val + std_val, alpha=0.2)

    plt.plot(epochs, avg_test, label=f'Test {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_test - std_test, avg_test + std_test, alpha=0.2)

    plt.plot(epochs, avg_mean, label=f'Mean {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_mean - std_mean, avg_mean + std_mean, alpha=0.2)

    plt.plot(epochs, avg_random, label=f'Random {score.value}', linewidth=2)
    plt.fill_between(
        epochs, avg_random - std_random, avg_random + std_random, alpha=0.2
    )

    plt.xlabel('Epoch')
    plt.ylabel(f'{score.value}')
    # if score == Scoring.mse:
    #     plt.yscale("log")
    plt.title(f'{model_name} : Average {score.value} Loss over Trials')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_avg_loss_r2(
    loss_tuple_run: List[List[Tuple[float, float, float]]],
    model_name: str,
    score: Scoring = Scoring.r2,
    save_filename: str = 'r2_plot.png',
) -> None:
    """Plots the averaged r2 over trials for train, validation, and test sets with std dev bands."""
    num_epochs = len(loss_tuple_run[0])

    data = np.array(loss_tuple_run)  # shape: (num_trials, num_epochs, 3)

    avg = data.mean(axis=0)  # shape: (num_epochs, 3)
    std = data.std(axis=0)  # shape: (num_epochs, 3)

    avg_train, avg_val, avg_test = (
        avg[:, 0],
        avg[:, 1],
        avg[:, 2],
    )
    std_train, std_val, std_test = (
        std[:, 0],
        std[:, 1],
        std[:, 2],
    )

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    plt.plot(epochs, avg_train, label=f'Train {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_train - std_train, avg_train + std_train, alpha=0.2)

    plt.plot(epochs, avg_val, label=f'Validation {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_val - std_val, avg_val + std_val, alpha=0.2)

    plt.plot(epochs, avg_test, label=f'Test {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_test - std_test, avg_test + std_test, alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel(f'{score.value}')
    plt.title(f'{model_name} : Average {score.value} Loss over Trials')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_avg_loss_with_baseline(
    loss_tuple_run: List[List[Tuple[float, float, float, float]]],
    model_name: str,
    score: Scoring = Scoring.mse,
    title: str | None = None,
    save_filename: str = 'mse_loss_plot.png',
) -> None:
    """Plots the averaged MSE loss over trials for train, validation, and test sets with std dev bands."""
    num_epochs = len(loss_tuple_run[0])

    data = np.array(loss_tuple_run)  # shape: (num_trials, num_epochs, 5)

    avg = data.mean(axis=0)  # shape: (num_epochs, 5)
    std = data.std(axis=0)  # shape: (num_epochs, 5)

    avg_train, avg_val, avg_test, avg_baseline = (
        avg[:, 0],
        avg[:, 1],
        avg[:, 2],
        avg[:, 3],
    )
    std_train, std_val, std_test, std_baseline = (
        std[:, 0],
        std[:, 1],
        std[:, 2],
        std[:, 3],
    )

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    plt.plot(epochs, avg_train, label=f'Train {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_train - std_train, avg_train + std_train, alpha=0.2)

    plt.plot(epochs, avg_val, label=f'Validation {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_val - std_val, avg_val + std_val, alpha=0.2)

    plt.plot(epochs, avg_test, label=f'Test {score.value}', linewidth=2)
    plt.fill_between(epochs, avg_test - std_test, avg_test + std_test, alpha=0.2)

    plt.plot(epochs, avg_baseline, label=f'Baseline {score.value}', linewidth=2)
    plt.fill_between(
        epochs, avg_baseline - std_baseline, avg_baseline + std_baseline, alpha=0.2
    )
    plt.xlabel('Epoch')
    plt.ylabel(f'{score.value}')
    if score == Scoring.mse:
        plt.yscale('log')
    if title is None:
        plt.title(f'{model_name} : Average {score.value} Loss over Trials')
    else:
        plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_all_loss_tuples(
    results_dir: str = 'results/logs',
) -> Dict[str, List[List[Tuple[float, float, float, float, float]]]]:
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
    all_results: Dict[str, List[List[Tuple[float, float, float, float, float]]]],
    metric: str = 'test',  # "train", "valid", or "test"
    save_filename: str = 'compare_models.png',
) -> None:
    """Plots the selected metric across models over epochs.

    Args:
        all_results: Dict from model_encoder to loss_tuple_run.
        metric: One of "train", "valid", or "test".
        save_filename: Name of the saved file (name of the png).
    """
    metric_index = {'train': 0, 'valid': 1, 'test': 2, 'mean': 3, 'random': 4}[metric]

    plt.figure(figsize=(10, 6))

    for label, loss_tuple_run in all_results.items():
        data = np.array(loss_tuple_run)  # shape: (runs, epochs, 5)
        avg_over_runs = data.mean(axis=0)  # shape: (epochs, 5)
        metric_values = avg_over_runs[:, metric_index]
        epochs = np.arange(1, len(metric_values) + 1)
        plt.plot(epochs, metric_values, label=label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.capitalize()} MSE')
    plt.yscale('log')
    plt.title(f'Comparison of {metric.capitalize()} MSE Across Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / 'combined'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    plt.savefig(save_path)
    plt.close()


def plot_model_per_encoder(
    all_results: dict[str, list[list[tuple[float, float, float, float, float]]]],
    metric: str = 'test',
    save_prefix: str = 'model_plot',
) -> None:
    """Plots the selected metric for each model across encoders over epochs.

    Args:
        all_results: Dict from model_encoder to loss_tuple_run.
        metric: One of "train", "valid", or "test".
        save_prefix: Prefix for saved plot filenames.
    """
    metric_index = {'train': 0, 'valid': 1, 'test': 2, 'mean': 3, 'random': 4}[metric]
    model_grouped: Dict[
        str, Dict[str, list[list[tuple[float, float, float, float, float]]]]
    ] = defaultdict(dict)  # {model: {encoder: loss_tuple_run}}

    for key, loss_tuple_run in all_results.items():
        try:
            model, encoder = key.split('_')
        except ValueError:
            print(f'Skipping unrecognized key format: {key}')
            continue
        model_grouped[model][encoder] = loss_tuple_run

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / 'combined'
    save_dir.mkdir(parents=True, exist_ok=True)

    for model, encoder_dict in model_grouped.items():
        plt.figure(figsize=(10, 6))
        for encoder, loss_tuple_run in encoder_dict.items():
            data = np.array(loss_tuple_run)  # shape: (runs, epochs, 5)
            avg_over_runs = data.mean(axis=0)  # shape: (epochs, 5)
            metric_values = avg_over_runs[:, metric_index]
            epochs = np.arange(1, len(metric_values) + 1)
            plt.plot(epochs, metric_values, label=encoder, linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} MSE')
        plt.yscale('log')
        plt.title(f'{model}: {metric.capitalize()} MSE per Encoder')
        plt.legend(title='Encoder')
        plt.grid(True)
        plt.tight_layout()

        save_path = save_dir / f'{save_prefix}_{model}_rmse_over_models.png'
        plt.savefig(save_path)
        plt.close()


def plot_metric_per_encoder(
    all_results: dict[str, list[list[tuple[float, float, float, float, float]]]],
    metric: str = 'test',
    save_prefix: str = 'encoder_plot',
) -> None:
    """Plots the selected metric for each encoder across models over epochs.

    Args:
        all_results: Dict from model_encoder to loss_tuple_run.
        metric: One of "train", "valid", or "test".
        save_prefix: Prefix for saved plot filenames.
    """
    metric_index = {'train': 0, 'valid': 1, 'test': 2, 'mean': 3, 'random': 4}[metric]
    encoder_grouped: Dict[
        str, Dict[str, list[list[tuple[float, float, float, float, float]]]]
    ] = defaultdict(dict)  # {model: {encoder: loss_tuple_run}}

    for key, loss_tuple_run in all_results.items():
        try:
            model, encoder = key.split('_')
        except ValueError:
            print(f'Skipping unrecognized key format: {key}')
            continue
        encoder_grouped[encoder][model] = loss_tuple_run

    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / 'combined'
    save_dir.mkdir(parents=True, exist_ok=True)

    for encoder, model_dict in encoder_grouped.items():
        plt.figure(figsize=(10, 6))
        for model, loss_tuple_run in model_dict.items():
            data = np.array(loss_tuple_run)  # shape: (runs, epochs, 3)
            avg_over_runs = data.mean(axis=0)  # shape: (epochs, 3)
            metric_values = avg_over_runs[:, metric_index]
            epochs = np.arange(1, len(metric_values) + 1)
            plt.plot(epochs, metric_values, label=model, linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} MSE')
        plt.yscale('log')
        plt.title(f'{encoder}: {metric.capitalize()} MSE per Model')
        plt.legend(title='Model')
        plt.grid(True)
        plt.tight_layout()

        save_path = save_dir / f'{save_prefix}_{encoder}.png'
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


def plot_spearman_pr_cr_correlation(heat_map: DataFrame) -> None:
    spearman_corr = heat_map.corr(method='spearman')

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'spearman_pr_cr_correlation_heatmap.png'

    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt='.4f',
        cmap='coolwarm',
        square=True,
        cbar_kws={'label': 'Spearman Correlation'},
    )
    plt.title('Spearman Correlation: PageRank vs Credibility')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f'Spearman correlation heatmap saved to: {save_path}')


def plot_pr_cr_scatter_logx(heat_map: DataFrame) -> None:
    heat_map = heat_map[heat_map['pr_value'] > 0]

    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pr_cr_scatter_logx.png'

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=heat_map, x='pr_value', y='cr_score', alpha=0.4, s=8)
    plt.xscale('log')
    plt.xlabel('PageRank (log scale)')
    plt.ylabel('Credibility Score')
    plt.title('PR vs CR Scatter Plot (Log X-Axis)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f'Scatter plot saved to: {save_path}')


def plot_pr_vs_cr_scatter(heat_map: pd.DataFrame) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'correlation' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'scatter_pr_vs_cr.png'

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=heat_map,
        x='pr_value',
        y='cr_score',
        alpha=0.3,
        edgecolor=None,
        s=10,
    )
    plt.title('Scatter Plot: PageRank vs Credibility Score')
    plt.xlabel('PageRank Value')
    plt.ylabel('Credibility Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f'Scatter plot saved to: {save_path}')


def plot_degree_distribution(degrees: list[int], experiment_name: str) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'topology' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.logspace(np.log10(1), np.log10(max(degrees) + 1), 50)
    ax.hist(degrees, bins=bins, color='steelblue', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{experiment_name} Degree Distribution (log-log)')
    plt.tight_layout()
    plt.savefig(save_dir / f'{experiment_name}_loghist.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(np.log10(degrees), fill=True, color='darkred', bw_adjust=0.5)
    ax.set_xlabel('log10(Degree)')
    ax.set_ylabel('Density')
    ax.set_title(f'{experiment_name} Degree KDE (log scale)')
    plt.tight_layout()
    plt.savefig(save_dir / f'{experiment_name}_kde.png')
    plt.close()


def plot_domain_scores(gov_scores: list[float], org_scores: list[float]) -> None:
    root = get_root_dir()
    save_dir = root / 'results' / 'topology' / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(gov_scores, fill=True, color='green', label='.gov', bw_adjust=0.5)
    sns.kdeplot(org_scores, fill=True, color='red', label='.org', bw_adjust=0.5)
    ax.set_xlabel('Domain PC1 Score')
    ax.set_ylabel('Density')
    ax.set_title('Domain PC1 Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'domain_pc1_kde.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gov_scores, bins=20, alpha=0.6, label='.gov', color='green')
    ax.hist(org_scores, bins=20, alpha=0.6, label='.org', color='red')
    ax.set_xlabel('Domain PC1 Score')
    ax.set_ylabel('Count')
    ax.set_title('Domain PC1 Score Histogram')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'domain_pc1_hist.png')
    plt.close()


def plot_avg_distribution(
    all_preds: List[Tensor], all_targets: List[Tensor], model_name: str, bins: int = 50
) -> None:
    """Plots the average distribution of predictions and targets."""
    root = get_root_dir()
    save_dir = root / 'results' / 'plots' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    preds = torch.cat(all_preds).detach().cpu().numpy()
    targets = torch.cat(all_targets).detach().cpu().numpy()

    avg_preds = np.mean(preds)
    avg_targets = np.mean(targets)

    plt.figure(figsize=(8, 5))
    plt.hist(preds, bins=bins, alpha=0.6, label=f'Predictions (avg={avg_preds:.4f})')
    plt.hist(targets, bins=bins, alpha=0.6, label=f'Targets (avg={avg_targets:.4f})')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Average Value Distributions: Predictions vs Targets')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_dir / 'train_pred_target_distribution.png')
