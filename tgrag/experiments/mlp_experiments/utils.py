from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import normalize 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sklearn.model_selection import KFold
def normalize_embeddings(emb_dict):
    norm_arr=normalize(list(emb_dict.values()))
    return {k: norm_arr[idx] for idx,k in enumerate(emb_dict.keys())}

def kfold_split(X, y, n_splits=5):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    splits = []
    for train_index, test_index in kf.split(X):
        X_tr_va, X_test = X.iloc[train_index], X.iloc[test_index]
        y_tr_va, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_tr_va, y_tr_va, test_size=0.25)
        splits.append((X_train, y_train,X_val, y_val, X_test, y_test))
    return splits
def train_valid_test_split(target, labeled_11k_df, test_valid_size=0.4):
    if target == "mbfc_bias":
        quantiles = labeled_11k_df[target].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        quantiles = labeled_11k_df[target].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
    bins = [labeled_11k_df[target].min()] + quantiles.tolist()
    labeled_11k_df[target + '_cat'] = pd.cut(labeled_11k_df[target], bins=bins, labels=quantiles, include_lowest=True)
    X = labeled_11k_df[['domain', target]]
    y = labeled_11k_df[target + '_cat']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_valid_size, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    return X_train, labeled_11k_df.iloc[X_train.index][target].tolist(), X_valid, labeled_11k_df.iloc[X_valid.index][
        target].tolist(), X_test, labeled_11k_df.iloc[X_test.index][target].tolist()


def resize_emb(text_emb, target, X_train, X_valid, X_test, gnn_emb=None, topic_emb=None, trim_to=1024):
    X_train_feat = [text_emb[d][0:trim_to] for d in X_train["domain"].tolist()]
    X_valid_feat = [text_emb[d][0:trim_to] for d in X_valid["domain"].tolist()]
    X_test_feat = [text_emb[d][0:trim_to] for d in X_test["domain"].tolist()]
    print("emb-size=", len(X_test_feat[0]))
    if gnn_emb is not None:
        X_train_feat_gnn = [gnn_emb[d][0:trim_to] for d in X_train["domain"].tolist()]
        X_train_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in zip(X_train_feat_gnn, X_train_feat)]

        X_valid_feat_gnn = [gnn_emb[d][0:trim_to] for d in X_valid["domain"].tolist()]
        X_valid_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in zip(X_valid_feat_gnn, X_valid_feat)]

        X_test_feat_gnn = [gnn_emb[d][0:trim_to] for d in X_test["domain"].tolist()]
        X_test_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in zip(X_test_feat_gnn, X_test_feat)]
        print("emb-size=", len(X_test_feat[0]))
    if topic_emb is not None:
        X_train_feat_topic = [topic_emb[d] for d in X_train["domain"].tolist()]
        X_train_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in
                        zip(X_train_feat_topic, X_train_feat)]

        X_valid_feat_topic = [topic_emb[d] for d in X_valid["domain"].tolist()]
        X_valid_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in
                        zip(X_valid_feat_topic, X_valid_feat)]

        X_test_feat_topic = [topic_emb[d] for d in X_test["domain"].tolist()]
        X_test_feat = [list(sublist1) + (list(sublist2)) for sublist1, sublist2 in zip(X_test_feat_topic, X_test_feat)]
        print("emb-size=", len(X_test_feat[0]))

    return X_train_feat, X_valid_feat, X_test_feat


def plot_loss(train_loss, valid_loss, test_loss, mean_loss, out_file_path="loss_plot.pdf"):
    plt.figure(figsize=(5, 4))
    plt.rc('font', size=16)
    plt.plot(range(len(train_loss)), train_loss, label="train loss")
    plt.plot(range(len(train_loss)), valid_loss, label="validation loss")
    plt.plot(range(len(train_loss)), test_loss, label="test loss")
    plt.plot(range(len(train_loss)), mean_loss, label="mean loss")
    plt.xticks(range(0, len(train_loss) + 1, 1 if len(train_loss) <= 10 else len(train_loss) // 10))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_histogram(true, pred, out_file_path="_testset_true_vs_pred_frequancy.pdf",target="pc1"):
    plt.figure(figsize=(5, 4))
    plt.hist(pred, bins=50, range=(0, 1), edgecolor='black', color='lightblue', label="Pred")
    plt.hist(true, bins=50, range=(0, 1), edgecolor='black', color='orange', alpha=0.6, label="True")
    y_max = max(np.histogram(true, bins=50, range=(0, 1))[0].max(),
                np.histogram(pred, bins=50, range=(0, 1))[0].max())
    plt.rc('font', size=15)
    plt.xticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.yticks(np.arange(0, y_max + 50, 100), rotation=0, ha='right')
    plt.xlabel(target.upper().replace("_", "-"))
    plt.ylabel('Frequancy')
    plt.legend()
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_regression_scatter(true, pred, out_file_path="_testset_true_vs_pred_scatter.pdf", target="pc1",):
    plt.figure(figsize=(5, 4))
    plt.rc('font', size=16)
    plt.scatter(true, pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], color='red', linestyle='-', label='regression line')
    plt.xticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.xlabel(target.upper().replace("_", "-"))
    plt.ylabel(f"Predicted {target.upper().replace('_', '-')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file_path,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def eval(pred, true):
    max(abs(x - y) for x, y in zip(true, pred))
    res_df = pd.DataFrame(zip(true, pred), columns=['true', 'pred'])
    res_df["diff"] = res_df.apply(lambda row: abs(row['true'] - row['pred']), axis=1)
    max_idx = res_df.idxmax()["diff"]
    min_idx = res_df.idxmin()["diff"]
    max_diff_row = res_df.iloc[max_idx]
    min_diff_row = res_df.iloc[min_idx]
    max_diff_dict = max_diff_row.to_dict()
    max_diff_dict["test_idx"] = max_idx
    min_diff_dict = min_diff_row.to_dict()
    min_diff_dict["test_idx"] = min_idx

    mse = mean_squared_error(true, pred)
    # print(f"mse={mse}")
    r2 = r2_score(true, pred)
    # print(f"r2={r2}")
    mae = mean_absolute_error(true, pred)
    true_mean = mean(true)
    mean_mae = mean_absolute_error(true, [true_mean for elem in true])
    # print(f"MAE={mae}")
    return mse, mae, r2, mean_mae, min_diff_dict, max_diff_dict
