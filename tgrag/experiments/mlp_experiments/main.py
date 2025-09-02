import argparse
import logging
from typing import Dict, cast
import pickle
import numpy as np
from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statistics import mean

def load_emb_dict(embed_type,path="../../../data"):
    if  embed_type=="text":
        with open(f'{path}/labeled_11k_scraped_text_emb.pkl', 'rb') as f:
            embd_dict=pickle.load(f)
    elif    embed_type=="domainName":
        with open(f'{path}/labeled_11k_domainname_emb.pkl', 'rb') as f:
            embd_dict=pickle.load(f)
    return embd_dict

def train_valid_test_split(target,labeled_11k_df,test_valid_size=0.4):
    if target == "pc1":
        quantiles = labeled_11k_df[target].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
    elif target == "mbfc_bias":
        quantiles = labeled_11k_df[target].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bins = [labeled_11k_df[target].min()] + quantiles.tolist()
    labeled_11k_df[target + '_cat'] = pd.cut(labeled_11k_df[target], bins=bins, labels=quantiles, include_lowest=True)
    X = labeled_11k_df[['domain', target]]
    y = labeled_11k_df[target + '_cat']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_valid_size, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    return X_train, labeled_11k_df.iloc[X_train.index][target].tolist(),  X_valid,labeled_11k_df.iloc[X_valid.index][target].tolist(), X_test, labeled_11k_df.iloc[X_test.index][target].tolist()

def resize_emb(embds,target,X_train,X_valid,X_test, trim_to = 1024):
    X_train_feat = [embds[d][0:trim_to] for d in X_train["domain"].tolist()]
    X_valid_feat = [embds[d][0:trim_to] for d in X_valid["domain"].tolist()]
    X_test_feat = [embds[d][0:trim_to] for d in X_test["domain"].tolist()]
    return X_train_feat, X_valid_feat, X_test_feat

def train_mlp(mlp_reg,X_train_feat,Y_train,X_valid_feat,Y_valid,X_test_feat,Y_test,epochs = 15 ):
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    for _ in tqdm(range(epochs)):
        for b in range(batch_size, len(Y_train), batch_size):
            X_batch, y_batch = X_train_feat[b - batch_size:b], Y_train[b - batch_size:b]
            batch_mean = sum(y_batch) / len(y_batch)
            mlp_reg.partial_fit(X_batch, y_batch)
            train_loss.append(mlp_reg.loss_)
            valid_loss.append(mean_squared_error(Y_valid, mlp_reg.predict(X_valid_feat)))
            test_loss.append(mean_squared_error(Y_test, mlp_reg.predict(X_test_feat)))
            mean_loss.append(mean_squared_error(y_batch, [batch_mean for elem in y_batch]))
    return mlp_reg,train_loss, valid_loss, test_loss, mean_loss

def plot_loss(train_loss, valid_loss, test_loss, mean_loss,out_path="../../visualizations",target="pc1",embed_type="text"):
    plt.figure(figsize=(5, 4))
    plt.rc('font', size=16)
    plt.plot(range(len(train_loss)), train_loss, label="train loss")
    plt.plot(range(len(train_loss)), valid_loss, label="validation loss")
    plt.plot(range(len(train_loss)), test_loss, label="test loss")
    plt.plot(range(len(train_loss)), mean_loss, label="mean loss")
    plt.xticks(range(0, len(train_loss) + 1, 2))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/11Kdataset_{target}_{embed_type}_loss_128-200-15-60-20-00.pdf", bbox_inches='tight',pad_inches=0.1)
    plt.show()
def plot_histogram(true,pred,out_path="../../visualizations",target="pc1",embed_type="text"):
    plt.figure(figsize=(5, 4))
    plt.hist(pred, bins=50, range=(0, 1), edgecolor='black', color='lightblue',label="Pred")
    plt.hist(true, bins=50, range=(0, 1), edgecolor='black', color='orange', alpha=0.6,label="True")
    y_max = max(np.histogram(true, bins=50, range=(0, 1))[0].max(),
                np.histogram(pred, bins=50, range=(0, 1))[0].max())
    plt.rc('font', size=15)
    plt.xticks(np.arange(0, 1.1, 0.2), rotation=0, ha='right')
    plt.yticks(np.arange(0, y_max + 50, 100), rotation=0, ha='right')
    plt.xlabel(target.upper().replace("_", "-"))
    plt.ylabel('Frequancy')
    plt.legend()
    plt.savefig(f"{out_path}/11Kdataset_{target}_{embed_type}_testset_true_vs_pred_frequancy.pdf",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
def plot_regression_scatter(true,pred,out_path="../../visualizations",target="pc1",embed_type="text"):
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
    plt.savefig(f"{out_path}/11Kdataset_{target}_{embed_type}_testset_true_vs_pred_scatter.pdf", bbox_inches='tight',pad_inches=0.1)
    plt.show()
def eval(pred,true):
    mse = mean_squared_error(true, pred)
    # print(f"mse={mse}")
    r2 = r2_score(true, pred)
    # print(f"r2={r2}")
    mae = mean_absolute_error(true, pred)
    # print(f"MAE={mae}")
    return mse,mae,r2
def main() -> None:
    root = get_root_dir()
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--target", type=str, default="mbfc_bias", choices=["pc1", "mbfc_bias"])
    parser.add_argument("--emb_path", type=str, default="data/dqr")
    parser.add_argument("--dqr_path", type=str, default="data/dqr")
    parser.add_argument("--embed_type", type=str, default="text", choices=["text", "domainName"])
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--test_valid_size", type=float, default=0.4)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--plots_out_path", type=str, default="plots")
    args = parser.parse_args()
    ############## Load training data and split ###############
    emb_dict=load_emb_dict(args.embed_type,args.emb_path)
    labeled_11k_df = pd.read_csv(f"{args.dqr_path}/domain_ratings.csv")
    X_train, y_train, X_valid, y_valid, X_test, y_test= train_valid_test_split(args.target, labeled_11k_df, args.test_valid_size)
    X_train_feat, X_valid_feat, X_test_feat=resize_emb(emb_dict,args.target,X_train,X_valid,X_test, args.emb_dim)
    ################# Train #####################
    mlp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',max_iter=args.max_iter, random_state=42, verbose=False, learning_rate_init=args.lr)
    mlp_reg,train_loss, valid_loss, test_loss, mean_loss=train_mlp(mlp_reg, X_train_feat, y_train, X_valid_feat, y_valid, X_test_feat, y_test,epochs=args.epochs)
    ################## Plot and Eval ###############
    true=y_test
    pred=mlp_reg.predict(X_test_feat)

    plot_loss(train_loss, valid_loss, test_loss, mean_loss,args.plots_out_path)
    plot_histogram(true,pred, args.plots_out_path)
    plot_regression_scatter(true,pred,args.plots_out_path)
    MSE,MAE,R2=eval(pred,true)
    print(f"MSE={MSE}\tR2={R2}\tMAE={MAE}")
if __name__ == '__main__':
    main()