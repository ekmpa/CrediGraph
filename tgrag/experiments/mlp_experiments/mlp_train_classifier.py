import argparse
import sklearn
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
from sklearn.neural_network import MLPClassifier as  Sklearn_MLPClassifier
from mlp_modules import MLPRegressor
from mlp_modules import MLP3LayersPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statistics import mean
from mlp_modules import MultiTaskMLP, train_multihead,train_mlp,train_scikitlearn_classifier,train_classifier,LabelPredictor,train_classifier_unbalanced
from sklearn.preprocessing import normalize 
from utils import normalize_embeddings, plot_histogram,train_valid_test_split,resize_emb,\
                  plot_histogram,plot_loss,plot_regression_scatter,eval,\
                  plot_classesCount
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from utils import search_parquet_emb,search_parquet_duckdb
def load_agg_Nmonth_emb_dict(embed_type, path="../../../data", model_name="embeddinggemma-300m",
                             month_lst=["dec", "nov", "oct"], target="pc1", agg="avg",normalize=False):
    months_emb_lst = []
    if embed_type == "GNN_GAT":
        for month in month_lst:
            with open(f'{path}/dqr_gnn_embeddings/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
                if normalize:
                    embd_dict=normalize_embeddings(embd_dict)
            months_emb_lst.append(embd_dict)
        common_domains_set = set(months_emb_lst[0].keys()).intersection(months_emb_lst[1].keys()).intersection(
            months_emb_lst[2].keys())

    for key in common_domains_set:
        for i in range(1, len(months_emb_lst)):
            if agg == "concat":
                months_emb_lst[0][key].extend(months_emb_lst[i][key])
            elif agg == "min":
                months_emb_lst[0][key] = [min(a, b) for a, b in zip(months_emb_lst[0][key], months_emb_lst[i][key])]
            elif agg == "max":
                months_emb_lst[0][key] = [max(a, b) for a, b in zip(months_emb_lst[0][key], months_emb_lst[i][key])]
            elif agg == "avg":
                months_emb_lst[0][key] = [(a + b) / 2 for a, b in zip(months_emb_lst[0][key], months_emb_lst[i][key])]

    concat_dict = {k: v for k, v in months_emb_lst[0].items() if k in common_domains_set}
    # print(f"concat Nmonth emb size={len(concat_dict[list(concat_dict.keys())[0]])}")
    # print(f"len of keys={len(concat_dict.keys())}")
    return concat_dict


def load_agg_Nmonth_weaksupervision_emb_dict(embed_type, path="../../../data", model_name="embeddinggemma-300m",
                                             month_lst=["dec", "nov", "oct"], target="pc1", agg="avg"):
    months_emb_PhishTank_lst = []
    months_emb_URLhaus_lst = []
    months_emb_legit_lst = []
    for month in month_lst:
        with open(f'{path}/PhishTank_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
            months_emb_PhishTank_lst.append(pickle.load(f))
        with open(f'{path}/URLHaus_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
            months_emb_URLhaus_lst.append(pickle.load(f))
        with open(f'{path}/IP2Location_{target}_rni_{month}_2024_embeddings.pkl', 'rb') as f:
            months_emb_legit_lst.append(pickle.load(f))

    for ds_months in [months_emb_PhishTank_lst, months_emb_URLhaus_lst, months_emb_legit_lst]:
        for key in ds_months[0].keys():
            for i in range(1, len(ds_months)):  # loop on dataset months
                if key in ds_months[i]:
                    if agg == "concat":
                        ds_months[0][key].extend(ds_months[i][key])
                        # print(len(ds_months[0][key]))
                    elif agg == "min":
                        ds_months[0][key] = [min(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                    elif agg == "max":
                        ds_months[0][key] = [max(a, b) for a, b in zip(ds_months[0][key], ds_months[i][key])]
                    elif agg == "avg":
                        ds_months[0][key] = [(a + b) / 2 for a, b in zip(ds_months[0][key], ds_months[i][key])]
                        # print(len(ds_months[0][key]))
    return months_emb_PhishTank_lst[0], months_emb_URLhaus_lst[0], months_emb_legit_lst[0]


def load_emb_dict(embed_type, path="../../../data",pickle_name=None, model_name="embeddinggemma-300m", month="dec", target="pc1", emb_dim=8192,normalize=False):
    if pickle_name:
        with open(f'{path}/{pickle_name}', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "text":
        if model_name == "embeddinggemma-300m":
            with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_{emb_dim}.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif model_name == "embeddingQwen3-0.6B":
            with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif model_name == "embeddingQwen3-8B":
            with open(f'{path}/dqr_{month}_text_embeddingQwen3-8B_4096.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif model_name == "embeddingTE3L":
            with open(f'{path}/dqr_{month}_text_embeddingTE3L_3072.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
    elif embed_type == "domainName":
        with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "GNN_GAT":
        # with open(f'{path}/11Kdataset_GAT_targets_connected_edges_GNN_textE_300E_pc1_emb.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        with open(f'{path}/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
            embd_dict = pickle.load(f)

    if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
        embd_dict={ k:v[0]['emb'][0:emb_dim] for k,v in embd_dict.items()}
    elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
        embd_dict={ k:v[0][1][0:emb_dim] for k,v in embd_dict.items()}
    if normalize:
        embd_dict=normalize_embeddings(embd_dict)
    return embd_dict

def load_emb_dict_from_parquet(embed_type, path="../../../data", model_name="embeddinggemma-300m", month="dec", target="pc1", emb_dim=8192,normalize=False):
    embd_dict=None
    if embed_type == "text":
        if model_name == "embeddinggemma-300m":
            # embd_dict=search_parquet_emb(path,f'weaksupervision_content_emb_{month}2024.parquet', column_name="domain",search_values=None,schema={'key':'domain','val':'embeddings'})
            embd_dict=search_parquet_duckdb(f'{path}/weaksupervision_content_emb_{month}2024.parquet', col="domain",q_domains=None,max_memory="8GB",schema={'key':'domain','val':'embeddings'})
    elif embed_type == "GNN_GAT":
        with open(f'{path}/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
            embd_dict = pickle.load(f)

    if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
        embd_dict={ k:v[0]['emb'][0:emb_dim] for k,v in embd_dict.items()}
    elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
        embd_dict={ k:v[0][1][0:emb_dim] for k,v in embd_dict.items()}
    if normalize:
        embd_dict=normalize_embeddings(embd_dict)
    return embd_dict



def mlp_classifier(args) -> None:
    ############## Load training data and split ###############
    full_emb_dict = load_emb_dict(args.embed_type, args.text_emb_path, pickle_name="weak_content_emb_embeddinggemma-300m_768.pkl",emb_dim=256,normalize=False)
    month_emb_dict = load_emb_dict_from_parquet(args.embed_type, args.text_emb_path, args.emb_model, args.month,normalize=False)
    weaklabeles_df = pd.read_csv(f"{args.weaksupervision_path}/weaklabels.csv")
    # targets_nodes_df = pd.read_csv(f"{args.dqr_path}/targets_nodes_df.csv")
    # targets_nodes_df["domain_rev"] = targets_nodes_df["domain"].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
    weaklabeles_df = weaklabeles_df[weaklabeles_df["domain"].isin(full_emb_dict)]
    weaklabeles_df = weaklabeles_df.reset_index(drop=True)
    text_emb_dict={}
    text_emb_dict.update(month_emb_dict)
    text_emb_dict.update({k:v for k,v in full_emb_dict.items() if k not in month_emb_dict})
    ############### filter by the GNN graph node splits ###################
    if args.filter_by_GNN_nodes:        
        test_domains_df=search_parquet_duckdb(f'{args.weaksupervision_path}/splits/{args.month}2024/test_split.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        test_domains_df['domain']=test_domains_df['Domains_test'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        print(f"test set lables count ={weaklabeles_df[weaklabeles_df["domain"].isin(test_domains_df['domain'])]["weak_label"].value_counts()}")
        valid_domains_df=search_parquet_duckdb(f'{args.weaksupervision_path}/splits/{args.month}2024/valid_split.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        valid_domains_df['domain']=valid_domains_df['Domains_valid'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        print(f"valid set lables count ={weaklabeles_df[weaklabeles_df["domain"].isin(valid_domains_df['domain'])]["weak_label"].value_counts()}")
        train_domains_df=search_parquet_duckdb(f'{args.weaksupervision_path}/splits/{args.month}2024/train_split.parquet', col=None,q_domains=None,max_memory="8GB",schema=None)
        train_domains_df['domain']=train_domains_df['Domains_train'].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
        print(f"train set lables count ={weaklabeles_df[weaklabeles_df["domain"].isin(train_domains_df['domain'])]["weak_label"].value_counts()}")
        test_domains_set=set(test_domains_df['domain']) 
        valid_domains_set=set(valid_domains_df['domain']) 
        train_domains_set=set(train_domains_df['domain']) 
        filter_by_domains_set=test_domains_set.union(valid_domains_set).union(train_domains_set)
        weaklabeles_df = weaklabeles_df[weaklabeles_df["domain"].isin(filter_by_domains_set)]
        weaklabeles_df = weaklabeles_df.reset_index(drop=True)
        text_emb_dict={k:v for k,v in text_emb_dict.items() if k in filter_by_domains_set}        
    ################# Train #####################
    results = []
    for i in range(args.runs):
        if args.filter_by_GNN_nodes: 
            X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.target, weaklabeles_df,key='domain',test_valid_size=args.test_valid_size,regressor=False,train_lst=train_domains_set,valid_lst=valid_domains_set,test_lst=test_domains_set)
        else:
            X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.target, weaklabeles_df,key='domain',test_valid_size=args.test_valid_size,regressor=False)
        ############### Resize embeddings ##################
        if args.embed_type == "TFIDF":
            print(f"TFIDF dim={text_emb_dict[list(text_emb_dict.keys())[0]].shape[0]}")
            X_train_feat, X_valid_feat, X_test_feat = resize_emb(text_emb_dict, args.target, X_train, X_valid, X_test,
                                                                 gnn_emb=None, trim_to=
                                                                 text_emb_dict[list(text_emb_dict.keys())[0]].shape[0])
        else:
            if args.use_gnn_emb or args.use_topic_emb:
                if args.use_gnn_emb:
                    if args.agg_month_emb:
                        gnn_emb_dict = load_agg_Nmonth_emb_dict("GNN_GAT", args.emb_path, agg="avg")
                    else:
                        gnn_emb_dict = load_emb_dict("GNN_GAT", args.gnn_emb_path,month=args.month)
                    weaklabeles_df = weaklabeles_df[weaklabeles_df["domain"].isin(gnn_emb_dict.keys())]
                    weaklabeles_df = weaklabeles_df.reset_index(drop=True)
                    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.target,weaklabeles_df,args.test_valid_size,regressor=False)
                else:
                    gnn_emb_dict = None
                if args.use_topic_emb:
                    # topic_iptc_emb_dict = load_emb_dict("IPTC_Topic", args.emb_path)
                    # topic_iptc_emb_dict = load_emb_dict("IPTC_Topic_freq", args.emb_path)
                    # topic_iptc_emb_dict = load_emb_dict("IPTC_Topic_emb", args.emb_path)
                    # topic_iptc_emb_dict = load_emb_dict("3Feat", args.emb_path)
                    # topic_iptc_emb_dict = load_emb_dict("3Feat2", args.emb_path)
                    # gnn_emb_dict = load_emb_dict("IPTC_Topic", args.emb_path)
                    topic_iptc_emb_dict = load_emb_dict("PASTEL_hasContent", args.text_emb_path)
                else:
                    topic_iptc_emb_dict = None
                X_train_feat, X_valid_feat, X_test_feat = resize_emb(text_emb_dict, args.target, X_train, X_valid,
                                                                     X_test, gnn_emb=gnn_emb_dict,
                                                                     topic_emb=topic_iptc_emb_dict,
                                                                     trim_to=args.emb_dim)
            else:
                X_train_feat, X_valid_feat, X_test_feat = resize_emb(text_emb_dict, args.target, X_train, X_valid,
                                                                     X_test, gnn_emb=None, topic_emb=None,
                                                                     trim_to=args.emb_dim)

        print(f"X_train_feat.shape={len(X_train_feat[0]) if type(X_train_feat[0]) == list else X_train_feat[0].shape}")
        if args.MultiHead == False:
            ###################### PYTorch Regressor  ######################
            if args.library=="pytorch":
                mlp_clf = LabelPredictor(len(X_train_feat[0]))
                # mlp_clf = MLPRegressor(len(X_train_feat[0]),hidden_layer_sizes=(int(len(X_train_feat[0]) * 0.75), 128, 64, 16))                
                mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_classifier_unbalanced(mlp_clf, X_train_feat, y_train,
                                                                                X_valid_feat, y_valid, X_test_feat,
                                                                                y_test, epochs=args.epochs)
                
                # mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_classifier(mlp_clf, X_train_feat, y_train,
                #                                                                 X_valid_feat, y_valid, X_test_feat,
                #                                                                 y_test, epochs=args.epochs)
            ######################## Scikit-Learn ###################
            elif args.library=="sklearn":
                mlp_clf = Sklearn_MLPClassifier(hidden_layer_sizes=(128, 32),
                                    activation='relu', solver='adam',max_iter=args.max_iter, random_state=42,
                                    verbose=False, learning_rate_init=args.lr,warm_start=True)
                mlp_clf.out_activation_ = 'sigmoid'
                mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_scikitlearn_classifier(mlp_clf, X_train_feat, y_train,
                                                                                X_valid_feat, y_valid, X_test_feat,
                                                                                y_test, epochs=args.epochs)
                                                                              
        else:
            mlp_clf = MultiTaskMLP(input_dim=len(X_train_feat[0]), num_classes=args.num_classes,
                                   hidden_dims=[int(len(X_train_feat[0]) * 0.9), 64, 16])            
            mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_multihead(mlp_clf, X_train_feat, y_train,
                                                                                    X_valid_feat, y_valid, X_test_feat,
                                                                                    y_test, lr=args.lr,
                                                                                    epochs=args.epochs, w_reg=0.5,
                                                                                    w_cls=0.5,
                                                                                    num_classes=args.num_classes)
        ###################### Save Model ####################
        run_file_name=f"{args.plots_out_path}/weaksupervision_{args.month}_{args.target}_{args.library}_{args.embed_type}_{args.emb_model}_{'GNN-RNI' if args.use_gnn_emb else ''}_{'topic-emb' if args.use_topic_emb else ''}_run{str(i)}"
        with open(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_clf, file)        
        ################## Plot and Eval ###############
        true = y_test
        pred = mlp_clf.predict(X_test_feat)
        pred=pred.round()
        accuracy = accuracy_score(true, pred)
        print(f"Accuracy: {accuracy:.4f}")
        cm = confusion_matrix(true, pred)
        print(f"cm: {cm}")        
        f1 = f1_score(true,pred, average='macro')
        print(f"F1 score (macro): {f1:.4f}")

        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf",ylabel="CrossEntropy Loss")
        plot_classesCount(cm,run_file_name+"_class_frequancy.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        MSE, MAE, R2, Mean_MAE, min_error_dict, max_error_dict = eval(pred, true)
        min_error_dict["domain"] = X_test.iloc[min_error_dict["test_idx"]]["domain"]
        max_error_dict["domain"] = X_test.iloc[max_error_dict["test_idx"]]["domain"]
        # results.append([MSE, MAE, R2, Mean_MAE, str(min_error_dict), str(max_error_dict), str(args)])
        # print(f"Run{i}:MSE={MSE}\tR2={R2}\tMAE={MAE}\tMean_MAE={Mean_MAE}\tmin_error_dict={min_error_dict}\tmax_error_dict={max_error_dict}")
        ############ save test results ############
        X_test.to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_testset.csv",index=None)
        pd.DataFrame(zip(true, pred), columns=["true", "pred"]).to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_pred.csv",index=None)

    results_df = pd.DataFrame(results, columns=['MSE', 'MAE', 'R2', 'Mean_MAE', 'Min_AE', 'Max_AE', 'args'])
    results_df.to_csv(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_results.csv",index=None)
    
    # for col in ['MSE', 'MAE', 'R2', 'Mean_MAE']:
    #     print(f"{col}: Mean={results_df[col].mean()}\tstd={results_df[col].std()}")


if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--target", type=str, default="weak_label", choices=["weak_label"], help="the credability target")
    parser.add_argument("--text_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--gnn_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--weaksupervision_path", type=str, default=str(root + "/data/weaksupervision"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="embeddinggemma-300m",
                        choices=["embeddingQwen3-8B", "embeddingQwen3-0.6B", "embeddinggemma-300m", "embeddingTE3L",
                                 "IPTC_Topic_emb"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=5000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=256,help="embedding size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=1e-2,help="learning rate")
    parser.add_argument("--epochs", type=int,default=200, help="# training epochs")  # default 20, 10 for Qwen38B, 10 for TFIDF-PC1, 4 for TFIDF-mbfc , 8 for TE3L, 7 for GNN+TE3L
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--use_gnn_emb", type=bool, default=False,help="append GNN embedding")
    parser.add_argument("--agg_month_emb", type=bool, default=False, help="aggregate montly GNN embeddings")
    parser.add_argument("--use_topic_emb", type=bool, default=False,help="use topic modeling features")
    parser.add_argument("--filter_by_GNN_nodes", type=bool, default=True,help="filter by domains has GNN embeddings")
    parser.add_argument("--MultiHead", type=bool, default=False, help="Use MultiHead Model")
    parser.add_argument("--num_classes", type=int, default=2,help="# classifcation classes for Multihead model")
    parser.add_argument("--generate_weaksupervision_scores", type=bool, default=False, help="gnerate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="nov", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    args = parser.parse_args()
    print("args=", args)
    mlp_classifier(args)