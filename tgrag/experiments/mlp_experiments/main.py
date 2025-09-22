import argparse
import logging
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import plot_histogram, plot_loss, plot_regression_scatter

parser = argparse.ArgumentParser(description='MLP Experiments')
parser.add_argument(
    '--target', type=str, default='mbfc_bias', choices=['pc1', 'mbfc_bias']
)
parser.add_argument('--emb_path', type=str, default='data/dqr')
parser.add_argument('--dqr_path', type=str, default='data/dqr')
parser.add_argument(
    '--embed_type', type=str, default='text', choices=['text', 'domainName']
)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--test_valid_size', type=float, default=0.4)
parser.add_argument('--emb_dim', type=int, default=1024)
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--plots_out_path', type=str, default='plots')
parser.add_argument(
    '--log-file',
    type=str,
    default='mlp_experiment.log',
    help='Name of log file at project root.',
)


def load_emb_dict(embed_type: str, path: str) -> Dict[str, np.ndarray]:
    if embed_type == 'text':
        with open(f'{path}/labeled_11k_scraped_text_emb.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == 'domainName':
        with open(f'{path}/labeled_11k_domainname_emb.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    return embd_dict


def train_valid_test_split(
    target: str, labeled_11k_df: pd.DataFrame, test_valid_size: float = 0.4
) -> Tuple[
    pd.DataFrame, List[float], pd.DataFrame, List[float], pd.DataFrame, List[float]
]:
    if target == 'pc1':
        quantiles = labeled_11k_df[target].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
    elif target == 'mbfc_bias':
        quantiles = labeled_11k_df[target].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bins = [labeled_11k_df[target].min()] + quantiles.tolist()
    labeled_11k_df[target + '_cat'] = pd.cut(
        labeled_11k_df[target], bins=bins, labels=quantiles, include_lowest=True
    )
    X = labeled_11k_df[['domain', target]]
    y = labeled_11k_df[target + '_cat']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_valid_size, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    return (
        X_train,
        labeled_11k_df.iloc[X_train.index][target].tolist(),
        X_valid,
        labeled_11k_df.iloc[X_valid.index][target].tolist(),
        X_test,
        labeled_11k_df.iloc[X_test.index][target].tolist(),
    )


def resize_emb(
    embds: Dict[str, np.ndarray],
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    trim_to: int = 1024,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    X_train_feat = [embds[d][0:trim_to] for d in X_train['domain'].tolist()]
    X_valid_feat = [embds[d][0:trim_to] for d in X_valid['domain'].tolist()]
    X_test_feat = [embds[d][0:trim_to] for d in X_test['domain'].tolist()]
    return X_train_feat, X_valid_feat, X_test_feat


def train_mlp(
    mlp_reg: MLPRegressor,
    X_train_feat: List[np.ndarray],
    Y_train: List[float],
    X_valid_feat: List[np.ndarray],
    Y_valid: List[float],
    X_test_feat: List[np.ndarray],
    Y_test: List[float],
    epochs: int = 15,
) -> Tuple[MLPRegressor, List[float], List[float], List[float], List[float]]:
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    for _ in tqdm(range(epochs)):
        for b in range(batch_size, len(Y_train), batch_size):
            X_batch, y_batch = (
                X_train_feat[b - batch_size : b],
                Y_train[b - batch_size : b],
            )
            batch_mean = sum(y_batch) / len(y_batch)
            mlp_reg.partial_fit(X_batch, y_batch)
            train_loss.append(mlp_reg.loss_)
            valid_loss.append(
                mean_squared_error(Y_valid, mlp_reg.predict(X_valid_feat))
            )
            test_loss.append(mean_squared_error(Y_test, mlp_reg.predict(X_test_feat)))
            mean_loss.append(
                mean_squared_error(y_batch, [batch_mean for elem in y_batch])
            )
    return mlp_reg, train_loss, valid_loss, test_loss, mean_loss


def eval(pred: np.ndarray, true: List[float]) -> Tuple[float, float, float]:
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    return mse, mae, r2


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    setup_logging(args.log_file)
    emb_dict = load_emb_dict(args.embed_type, root / args.emb_path)
    labeled_11k_df = pd.read_csv(root / 'dqr' / 'domain_ratings.csv')
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(
        args.target, labeled_11k_df, args.test_valid_size
    )
    X_train_feat, X_valid_feat, X_test_feat = resize_emb(
        emb_dict, X_train, X_valid, X_test, args.emb_dim
    )
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=args.max_iter,
        random_state=42,
        verbose=False,
        learning_rate_init=args.lr,
    )
    mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_mlp(
        mlp_reg,
        X_train_feat,
        y_train,
        X_valid_feat,
        y_valid,
        X_test_feat,
        y_test,
        epochs=args.epochs,
    )
    true = y_test
    pred = mlp_reg.predict(X_test_feat)

    plot_loss(train_loss, valid_loss, test_loss, mean_loss, args.plots_out_path)
    plot_histogram(true, pred, args.plots_out_path)
    plot_regression_scatter(true, pred, args.plots_out_path)
    MSE, MAE, R2 = eval(pred, true)
    logging.info(f'MSE={MSE}\tR2={R2}\tMAE={MAE}')


if __name__ == '__main__':
    main()
