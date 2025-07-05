import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
