import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

from tgrag.dataset.temporal_dataset import TemporalDataset
from tgrag.gnn.model import Model
from tgrag.utils.args import DataArguments, ModelArguments
from tgrag.utils.logger import Logger
from tgrag.utils.plot import (
    Scoring,
    mean_across_lists,
    plot_avg_loss,
    plot_avg_loss_r2,
    plot_pred_target_distributions_bin_list,
)
from tgrag.utils.prob import ragged_mean_by_index
from tgrag.utils.save import save_loss_results


def _extract_mid_predictions(preds: Tensor, prediction_dim: int) -> Tensor:
    if prediction_dim == 1:
        return preds.squeeze(-1)
    return preds[:, 0]


def _quantile_loss(
    preds: Tensor,
    targets: Tensor,
    mask: Tensor,
    alpha: float,
) -> Tensor:
    mid = preds[:, 0]
    lower = preds[:, 1]
    upper = preds[:, 2]
    labels = targets
    mid_vals = mid[mask]
    lower_vals = lower[mask]
    upper_vals = upper[mask]
    label_vals = labels[mask]
    if label_vals.numel() == 0:
        return torch.tensor(0.0, device=preds.device)
    mse_loss = F.mse_loss(mid_vals, label_vals)
    low_bound = alpha / 2
    upp_bound = 1 - alpha / 2
    low_loss = torch.mean(
        torch.max(
            (low_bound - 1) * (label_vals - lower_vals),
            low_bound * (label_vals - lower_vals),
        )
    )
    upp_loss = torch.mean(
        torch.max(
            (upp_bound - 1) * (label_vals - upper_vals),
            upp_bound * (label_vals - upper_vals),
        )
    )
    ordering_loss = torch.mean(torch.relu(lower_vals - upper_vals))
    return mse_loss + low_loss + upp_loss + ordering_loss


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
    prediction_dim: int,
    use_quantile_heads: bool,
    quantile_alpha: float,
) -> Tuple[float, float, Tensor, Tensor]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        mid_preds = _extract_mid_predictions(preds, prediction_dim)
        targets = batch.y
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        if use_quantile_heads:
            loss = _quantile_loss(preds, targets, train_mask, quantile_alpha)
        else:
            loss = F.l1_loss(mid_preds[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        all_preds.append(mid_preds[train_mask])
        all_targets.append(targets[train_mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    avg_preds = ragged_mean_by_index(all_preds)
    avg_targets = ragged_mean_by_index(all_targets)
    mse = total_loss / total_batches
    return (mse, r2, avg_preds, avg_targets)


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
    prediction_dim: int,
    use_quantile_heads: bool,
    quantile_alpha: float,
) -> Tuple[float, float, List[float], List[float]]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    # TODO: Score in one list
    pred_scores = []
    target_scores = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        mid_preds = _extract_mid_predictions(preds, prediction_dim)
        targets = batch.y
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        if use_quantile_heads:
            loss = _quantile_loss(preds, targets, train_mask, quantile_alpha)
        else:
            loss = F.l1_loss(mid_preds[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        all_preds.append(mid_preds[train_mask])
        all_targets.append(targets[train_mask])
        for pred in mid_preds[train_mask]:
            pred_scores.append(pred.item())
        for targ in targets[train_mask]:
            target_scores.append(targ.item())

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    ragged_mean_by_index(all_preds)
    ragged_mean_by_index(all_targets)
    mse = total_loss / total_batches
    return (mse, r2, pred_scores, target_scores)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
    prediction_dim: int,
) -> Tuple[float, float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        mid_preds = _extract_mid_predictions(preds, prediction_dim)
        targets = batch.y
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        # MEAN: 0.546
        mean_preds = torch.full(batch.y[mask].size(), 0.5).to(device)
        random_preds = torch.rand(batch.y[mask].size(0)).to(device)
        loss = F.l1_loss(mid_preds[mask], targets[mask])
        mean_loss = F.l1_loss(mean_preds, targets[mask])
        random_loss = F.l1_loss(random_preds, targets[mask])

        # TODO: Change this to report the loss of mean to be accurate. Use full score for don't average per batch.
        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        total_batches += 1

        all_preds.append(mid_preds[mask])
        all_targets.append(targets[mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    mse = total_loss / total_batches
    mse_mean = total_mean_loss / total_batches
    mse_random = total_random_loss / total_batches
    return (mse, mse_mean, mse_random, r2)


def run_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalDataset,
    experiment_name: Optional[str] = None,
    target_col: Optional[str] = None,
) -> None:
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logging.info(f'Device found: {device}')

    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Train loader created')

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    logging.info('Valid loader created')
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Test loader created')

    logger = Logger(model_arguments.runs)
    loss_tuple_run_mse: List[List[Tuple[float, float, float, float, float]]] = []
    loss_tuple_run_r2: List[List[Tuple[float, float, float]]] = []
    final_avg_preds: List[List[float]] = []
    final_avg_targets: List[List[float]] = []
    global_best_val_loss = float('inf')
    best_state_dict = None
    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            prediction_dim=model_arguments.prediction_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        loss_tuple_epoch_r2: List[Tuple[float, float, float]] = []
        epoch_avg_preds: List[List[float]] = []
        epoch_avg_targets: List[List[float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            _, _, batch_preds, batch_targets = train_(
                model,
                train_loader,
                optimizer,
                model_arguments.prediction_dim,
                model_arguments.use_quantile_heads,
                model_arguments.quantile_alpha,
            )
            epoch_avg_preds.append(batch_preds)
            epoch_avg_targets.append(batch_targets)
            train_loss, _, _, train_r2 = evaluate(
                model,
                train_loader,
                'train_mask',
                model_arguments.prediction_dim,
            )
            valid_loss, valid_mean_baseline_loss, _, valid_r2 = evaluate(
                model,
                val_loader,
                'valid_mask',
                model_arguments.prediction_dim,
            )
            test_loss, test_mean_baseline_loss, test_random_baseline_loss, test_r2 = (
                evaluate(
                    model,
                    test_loader,
                    'test_mask',
                    model_arguments.prediction_dim,
                )
            )
            result = (
                train_loss,
                valid_loss,
                test_loss,
                test_mean_baseline_loss,
                test_random_baseline_loss,
            )
            result_r2 = (train_r2, valid_r2, test_r2)
            loss_tuple_epoch_mse.append(result)
            loss_tuple_epoch_r2.append(result_r2)
            logger.add_result(
                run, (train_loss, valid_loss, test_loss, valid_mean_baseline_loss)
            )
            if valid_loss < global_best_val_loss:
                global_best_val_loss = valid_loss
                best_state_dict = model.state_dict()

        final_avg_preds.append(mean_across_lists(epoch_avg_preds))
        final_avg_targets.append(mean_across_lists(epoch_avg_targets))
        loss_tuple_run_mse.append(loss_tuple_epoch_mse)
        loss_tuple_run_r2.append(loss_tuple_epoch_r2)

    best_model_dir = weight_directory / f'{model_arguments.model}'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / 'best_model.pt'
    torch.save(best_state_dict, best_model_path)
    logging.info(f'Model: {model_arguments} weights saved to: {best_model_path}')
    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics())
    logging.info(logger.get_avg_statistics())
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=10
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=5
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=1
        )
    )
    logging.info('Constructing plots')
    plot_pred_target_distributions_bin_list(
        preds=final_avg_preds,
        targets=final_avg_targets,
        model_name=model_arguments.model,
        bins=100,
    )
    plot_avg_loss(
        loss_tuple_run_mse, model_arguments.model, Scoring.mae, 'loss_plot.png'
    )
    plot_avg_loss_r2(
        loss_tuple_run_r2, model_arguments.model, Scoring.r2, 'r2_plot.png'
    )
    logging.info('Saving pkl of results')
    # Use experiment_name if provided, otherwise fall back to model name
    # This distinguishes between Q_GAT (quantile) and GAT (baseline)
    save_model_name = (
        experiment_name if experiment_name is not None else model_arguments.model
    )
    # Include target_col to distinguish PC1 and MBFC results
    # target_col is passed from main.py (from meta_args) or can be obtained from dataset
    save_target_col = target_col if target_col is not None else dataset.target_col
    save_loss_results(loss_tuple_run_mse, save_model_name, 'TODO', save_target_col)
