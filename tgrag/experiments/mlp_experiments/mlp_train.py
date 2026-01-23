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
from sklearn.neural_network import MLPRegressor as  Sklearn_MLPRegressor
from mlp_modules import MLPRegressor
from mlp_modules import MLP3LayersPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statistics import mean
from mlp_modules import MultiTaskMLP, train_multihead,train_mlp,train_scikitlearn_regressor
from sklearn.preprocessing import normalize 
from utils import normalize_embeddings,train_valid_test_split,resize_emb,plot_histogram,plot_loss,plot_regression_scatter,eval

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


def load_emb_dict(embed_type, path="../../../data", model_name="embeddinggemma-300m", month="dec", target="pc1", emb_dim=256,normalize=False):
    if embed_type == "text":
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
        elif model_name == "IPTC_Topic_emb":
            with open(f'IPTCTopicModeling/dqr_dec_IPTC_predFinalLayer_emb_dict.pkl','rb') as f:
                embd_dict = pickle.load(f)
    elif embed_type == "domainName":
        with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "GNN_GAT":
        # with open(f'{path}/11Kdataset_GAT_targets_connected_edges_GNN_textE_300E_pc1_emb.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        with open(f'{path}/{month}_{target}_dqr_domain_rni_embeddings.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "IPTC_Topic":
        with open(f'IPTCTopicModeling/dqr_IPTC-news-topic_scores.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "IPTC_Topic_freq":
        with open(f'IPTCTopicModeling/dqr_topics_frequancy_norm_dict.pkl','rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "IPTC_Topic_emb":
        with open(f'IPTCTopicModeling/dqr_dec_IPTC_predFinalLayer_emb_dict.pkl','rb') as f:
            embd_dict = pickle.load(f)
        # print(list(embd_dict.keys())[0],embd_dict[list(embd_dict.keys())[0]])
    elif embed_type == "3Feat":
        with open(f'/shared_mnt/github_repos/CrediGraph/data/dqr/dqr_3Feat_dict.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "3Feat2":
        with open(f'/shared_mnt/github_repos/CrediGraph/data/dqr/dqr_3Feat_dict2.pkl', 'rb') as f:
            embd_dict = pickle.load(f)

    elif embed_type == "TFIDF":
        # with open(f'{path}/dqr_TFIDF_emb.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        # with open(f'{path}/dqr_TFIDF_emb_8465.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        # with open(f'{path}/dqr_TFIDF_emb_19437.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        if month == "dec":
            with open(f'{path}/dqr_dec_TFIDF_weaksupervision_emb_222755.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif month == "nov":
            with open(f'{path}/dqr_nov_TFIDF_weaksupervision_emb_258729.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif month == "oct":
            with open(f'{path}/dqr_oct_TFIDF_weaksupervision_emb_19085.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
    elif embed_type == "PASTEL":
        with open(f'{path}/dqr_pastel_dict.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "PASTEL_hasContent":
        with open(f'{path}/dqr_hasContent_pastel_dict.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    if isinstance(embd_dict[list(embd_dict.keys())[0]][0], dict): #list of dicts per domain pages (parquet format)
        embd_dict={ k:v[0]['emb'] for k,v in embd_dict.items()}
    elif isinstance(embd_dict[list(embd_dict.keys())[0]][0], list) and isinstance(embd_dict[list(embd_dict.keys())[0]][0][0], str) : #list of lists per domain pages (parquet format)
        embd_dict={ k:v[0][1] for k,v in embd_dict.items()}
    if normalize:
        embd_dict=normalize_embeddings(embd_dict)
    return embd_dict


def load_weaksupervision_emb_dict(embed_type, path="../../../data", model_name="embeddinggemma-300m", month="dec",
                                  target="pc1", gnn_emb=None, agg=None):
    embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = {}, {}, {}
    if embed_type == "text":
        if model_name == "embeddinggemma-300m":
            with open(f'{path}/dqr_{month}_text_embeddinggemma-300m_768.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif model_name == "embeddingQwen3-0.6B":
            with open(f'{path}/dqr_{month}_text_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
                embd_dict = pickle.load(f)
        elif model_name == "embeddingQwen3-8B":
            with open(f'{path}/cc_dec_2024_phishtank_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                embd_dict_phishtank = pickle.load(f)
            with open(f'{path}/cc_dec_2024_URLhaus_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                embd_dict_URLhaus = pickle.load(f)
            with open(f'{path}/cc_dec_2024_PhishDataset_legit_Qwen3-Embedding-8B_4096.pkl', 'rb') as f:
                embd_dict_PhishDataset_legit = pickle.load(f)
        elif model_name == "embeddingTE3L":
            with open(f'{path}/phishtank_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                embd_dict_phishtank = pickle.load(f)
            with open(f'{path}/URLhaus_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                embd_dict_URLhaus = pickle.load(f)
            with open(f'{path}/PhishDataset_legit_{month}_TE3L_weaksupervision_emb_3072.pkl', 'rb') as f:
                embd_dict_PhishDataset_legit = pickle.load(f)

        if gnn_emb == True:
            print("len of embd_dict_phishtank before appending GNN=",
                  len(embd_dict_phishtank[list(embd_dict_phishtank.keys())[0]]))
            if agg is None:
                with open(f'{path}/PhishTank_{target}_rni_embeddings.pkl', 'rb') as f:
                    gnn_embd_dict_phishtank = pickle.load(f)
                with open(f'{path}/URLHaus_{target}_rni_embeddings.pkl', 'rb') as f:
                    gnn_embd_dict_URLhaus = pickle.load(f)
                with open(f'{path}/IP2Location_{target}_rni_embeddings.pkl', 'rb') as f:
                    gnn_embd_dict_PhishDataset_legit = pickle.load(f)
            else:
                gnn_embd_dict_phishtank, gnn_embd_dict_URLhaus, gnn_embd_dict_PhishDataset_legit = load_agg_Nmonth_weaksupervision_emb_dict(
                    embed_type, path, model_name, month_lst=["dec", "nov", "oct"], target=target, agg=agg)

            ############# Append Emb #############
            for k in embd_dict_phishtank:
                embd_dict_phishtank[k] = gnn_embd_dict_phishtank[k] + embd_dict_phishtank[k]
            for k in embd_dict_URLhaus:
                embd_dict_URLhaus[k] = gnn_embd_dict_URLhaus[k] + embd_dict_URLhaus[k]
            for k in embd_dict_PhishDataset_legit:
                embd_dict_PhishDataset_legit[k] = gnn_embd_dict_PhishDataset_legit[k] + embd_dict_PhishDataset_legit[k]
            print("len of embd_dict_phishtank after appending GNN=",
                  len(embd_dict_phishtank[list(embd_dict_phishtank.keys())[0]]))


    elif embed_type == "domainName":
        with open(f'{path}/dqr_domainName_embeddingQwen3-0.6B_1024.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "GNN_GAT":
        with open(f'{path}/11Kdataset_GAT_targets_connected_edges_GNN_textE_300E_pc1_emb.pkl', 'rb') as f:
            embd_dict = pickle.load(f)
    elif embed_type == "TFIDF":
        # with open(f'{path}/dqr_TFIDF_emb.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        # with open(f'{path}/dqr_TFIDF_emb_8465.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        # with open(f'{path}/dqr_dec_TFIDF_emb_19437.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        # with open(f'{path}/dqr_TFIDF_emb_19437.pkl', 'rb') as f:
        #     embd_dict=pickle.load(f)
        emb_size = "222755" if month == "dec" else "258729" if month == "nov" else "19085"
        with open(f'{path}/phishtank_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
            embd_dict_phishtank = pickle.load(f)
        with open(f'{path}/URLhaus_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
            embd_dict_URLhaus = pickle.load(f)
        with open(f'{path}/phishDataset_legit_{month}_TFIDF_weaksupervision_emb_{emb_size}.pkl', 'rb') as f:
            embd_dict_PhishDataset_legit = pickle.load(f)

    return embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit

def mlp_train(args) -> None:
    ############## Load training data and split ###############
    text_emb_dict = load_emb_dict(args.embed_type, args.text_emb_path, args.emb_model, args.month,normalize=False)
    labeled_11k_df = pd.read_csv(f"{args.dqr_path}/domain_ratings.csv")
    targets_nodes_df = pd.read_csv(f"{args.dqr_path}/targets_nodes_df.csv")
    targets_nodes_df["domain_rev"] = targets_nodes_df["domain"].apply(lambda x: '.'.join(str(x).split('.')[::-1]))
    labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(text_emb_dict)]
    labeled_11k_df = labeled_11k_df.reset_index(drop=True)
    ############### filter by the 8K GNN Nodes ###################
    if args.filter_by_GNN_nodes:        
        labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(targets_nodes_df["domain_rev"])]
        labeled_11k_df = labeled_11k_df.reset_index(drop=True)
    if args.filter_by_PASTEL_domains:
        pastel_emb_dict = {}
        with open(f'{args.dqr_path}/dqr_hasContent_pastel_dict.pkl', 'rb') as f:
            pastel_emb_dict = pickle.load(f)
        labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(pastel_emb_dict.keys())]
        labeled_11k_df = labeled_11k_df.reset_index(drop=True)
    ################# Train #####################
    results = []
    for i in range(args.runs):
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.target, labeled_11k_df,args.test_valid_size)
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
                    labeled_11k_df = labeled_11k_df[labeled_11k_df["domain"].isin(gnn_emb_dict.keys())]
                    labeled_11k_df = labeled_11k_df.reset_index(drop=True)
                    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(args.target,labeled_11k_df,args.test_valid_size)
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
                mlp_reg = MLP3LayersPredictor(len(X_train_feat[0]))
                # mlp_reg = MLPRegressor(len(X_train_feat[0]),hidden_layer_sizes=(int(len(X_train_feat[0]) * 0.75), 128, 64, 16))
                mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_mlp(mlp_reg, X_train_feat, y_train,
                                                                                X_valid_feat, y_valid, X_test_feat,
                                                                                y_test, epochs=args.epochs)
            ######################## Scikit-Learn ###################
            elif args.library=="sklearn":
                mlp_reg = Sklearn_MLPRegressor(hidden_layer_sizes=(128, 32),
                                    activation='relu', solver='adam',max_iter=args.max_iter, random_state=42,
                                    verbose=False, learning_rate_init=args.lr)
                mlp_reg.out_activation_ = 'sigmoid'
                mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_scikitlearn_regressor(mlp_reg, X_train_feat, y_train,
                                                                                X_valid_feat, y_valid, X_test_feat,
                                                                                y_test, epochs=args.epochs)
                                                                              
        else:
            mlp_reg = MultiTaskMLP(input_dim=len(X_train_feat[0]), num_classes=args.num_classes,
                                   hidden_dims=[int(len(X_train_feat[0]) * 0.9), 64, 16])            
            mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_multihead(mlp_reg, X_train_feat, y_train,
                                                                                    X_valid_feat, y_valid, X_test_feat,
                                                                                    y_test, lr=args.lr,
                                                                                    epochs=args.epochs, w_reg=0.5,
                                                                                    w_cls=0.5,
                                                                                    num_classes=args.num_classes)
        ###################### Save Model ####################
        run_file_name=f"{args.plots_out_path}/dqr_{args.month}_{args.target}_{args.library}_{args.embed_type}_{args.emb_model}_{'GNN-RNI' if args.use_gnn_emb else ''}_{'topic-emb' if args.use_topic_emb else ''}_run{str(i)}"
        with open(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_reg, file)
        ################## Weak Supervision ###########
        if args.generate_weaksupervision_scores == True:
            if args.agg_month_emb:
                embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = load_weaksupervision_emb_dict(
                    args.embed_type, args.text_emb_path, args.emb_model, month="dec", target=args.target,
                    gnn_emb=args.use_gnn_emb, agg="avg")
            else:
                embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = load_weaksupervision_emb_dict(
                    args.embed_type, args.text_emb_path, args.emb_model, month="dec", target=args.target,
                    gnn_emb=args.use_gnn_emb, agg=None)

            phishtank_features = [v for k, v in embd_dict_phishtank.items()]
            phishtank_pred = mlp_reg.predict(phishtank_features)
            pd.DataFrame(zip(embd_dict_phishtank.keys(), phishtank_pred),
                         columns=["domain", f"pred_{args.target}"]).to_csv(
                        f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_Phishtank_pred.csv",index=None)
            # print("phishtank_pred=",phishtank_pred)

            URLhaus_features = [v for k, v in embd_dict_URLhaus.items()]
            URLhaus_pred = mlp_reg.predict(URLhaus_features)
            pd.DataFrame(zip(embd_dict_URLhaus.keys(), URLhaus_pred), columns=["domain", f"pred_{args.target}"]).to_csv(
                f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_URLhaus_pred.csv",index=None)

            PhishDataset_legit_features = [v for k, v in embd_dict_PhishDataset_legit.items()]
            PhishDataset_legit_pred = mlp_reg.predict(PhishDataset_legit_features)
            pd.DataFrame(zip(embd_dict_PhishDataset_legit.keys(), PhishDataset_legit_pred),
                         columns=["domain", f"pred_{args.target}"]).to_csv(
                        f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_legit_pred.csv",index=None)
        ################## Plot and Eval ###############
        true = y_test
        pred = mlp_reg.predict(X_test_feat) 
        
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf")
        plot_histogram(true, pred,  run_file_name+"_testset_true_vs_pred_frequancy.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        MSE, MAE, R2, Mean_MAE, min_error_dict, max_error_dict = eval(pred, true)
        min_error_dict["domain"] = X_test.iloc[min_error_dict["test_idx"]]["domain"]
        max_error_dict["domain"] = X_test.iloc[max_error_dict["test_idx"]]["domain"]
        results.append([MSE, MAE, R2, Mean_MAE, str(min_error_dict), str(max_error_dict), str(args)])
        print(
            f"Run{i}:MSE={MSE}\tR2={R2}\tMAE={MAE}\tMean_MAE={Mean_MAE}\tmin_error_dict={min_error_dict}\tmax_error_dict={max_error_dict}")
        ############ save test results ############
        X_test.to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_testset.csv",index=None)
        pd.DataFrame(zip(true, pred), columns=["true", "pred"]).to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_pred.csv",index=None)

    results_df = pd.DataFrame(results, columns=['MSE', 'MAE', 'R2', 'Mean_MAE', 'Min_AE', 'Max_AE', 'args'])
    results_df.to_csv(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_results.csv",index=None)
    
    for col in ['MSE', 'MAE', 'R2', 'Mean_MAE']:
        print(f"{col}: Mean={results_df[col].mean()}\tstd={results_df[col].std()}")


if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--target", type=str, default="pc1", choices=["pc1", "mbfc", "mbfc_bias"], help="the credability target")
    parser.add_argument("--text_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--gnn_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--dqr_path", type=str, default=str(root + "/data/dqr"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="embeddingTE3L",
                        choices=["embeddingQwen3-8B", "embeddingQwen3-0.6B", "embeddinggemma-300m", "embeddingTE3L",
                                 "IPTC_Topic_emb"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=5000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=4096,help="embedding size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=5e-3,help="learning rate")
    parser.add_argument("--epochs", type=int,default=100, help="# training epochs")  # default 20, 10 for Qwen38B, 10 for TFIDF-PC1, 4 for TFIDF-mbfc , 8 for TE3L, 7 for GNN+TE3L
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--use_gnn_emb", type=bool, default=True,help="append GNN embedding")
    parser.add_argument("--agg_month_emb", type=bool, default=False, help="aggregate montly GNN embeddings")
    parser.add_argument("--use_topic_emb", type=bool, default=False,help="use topic modeling features")
    parser.add_argument("--filter_by_GNN_nodes", type=bool, default=True,help="filter by domains has GNN embeddings")
    parser.add_argument("--filter_by_PASTEL_domains", type=bool, default=False,help="filter by domains has PASTEL embeddings")
    parser.add_argument("--MultiHead", type=bool, default=False, help="Use MultiHead Model")
    parser.add_argument("--num_classes", type=int, default=3,help="# classifcation classes for Multihead model")
    parser.add_argument("--generate_weaksupervision_scores", type=bool, default=False, help="gnerate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="sklearn", choices=["pytorch", "sklearn"],help="ML library to use")
    args = parser.parse_args()
    print("args=", args)
    mlp_train(args)