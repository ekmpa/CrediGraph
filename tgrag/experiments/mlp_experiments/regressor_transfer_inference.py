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
import pyarrow.parquet as pq
import pyarrow.compute as pc
import multiprocessing
from multiprocessing import Pool, cpu_count
import threading
from utils import search_parquet_duckdb
def load_emb_dict(emb_pickle_path="domain_shard_index.pkl"):       
    embd_dict=None
    with open(emb_pickle_path, 'rb') as f:
        embd_dict = pickle.load(f)
    return embd_dict

def write_results(labels_annot_df,emb_file_name):
    sources=["all_sources","legit-phish","misinfo-domains","nelez","phish-and-legit","phish-dataset","url-phish","urlhaus","wikipedia"]
    acc_results=[]
    for source in sources:
        res=[]
        res.append(source)
        if source=="all_sources":
            source_df=labels_annot_df
        else:
            source_df=labels_annot_df[~labels_annot_df[source].isna()]
        res.append(len(source_df))
        for th in [0.1,0.3,0.5,0.7,0.9]:
            k=f"pred_label_gte_{int(th*10)}"
            filtered_df = source_df[source_df['binary_label'] == source_df[k]]
            res.append(filtered_df[k].sum()/len(source_df))
        acc_results.append(res)
    
    res_df=pd.DataFrame(acc_results,columns=["source","count","gte_1","gte_3","gte_5","gte_7","gte_9"])
    res_df.to_csv(f"{emb_file_name.replace("_test_set_emb_dict.pkl","_ACC_results.csv")}",index=None)

def main() -> None:
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP transfer Inference")
    parser.add_argument("--emb_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/plots") ,help="text emb files path")    
    # parser.add_argument("--emb_file_name", type=str, default=str("weaksupervision_dec_text_TE3L_test_set_emb_dict.pkl") ,help="emb files name")        
    parser.add_argument("--emb_file_name", type=str, default=str("weaksupervision_dec_text_TE3L_GAT-text-avg_test_set_emb_dict.pkl") ,help="emb files name")   

    parser.add_argument("--model_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/plots") ,help="text emb files path")    
    # parser.add_argument("--model_file_name", type=str, default=str("dqr_dec_pc1_sklearn_text_embeddingTE3L__run1_credibench_MLP_Model.pkl") ,help="trained model name")
    parser.add_argument("--model_file_name", type=str, default=str("dqr_dec_pc1_sklearn_text_embeddingTE3L_GAT-text-avg_run1_agg_credibench_MLP_Model.pkl") ,help="trained model name")

    parser.add_argument("--labels_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/data/weaksupervision") ,help="text emb files path")    
    parser.add_argument("--labels_file_name", type=str, default=str("weaklabels.csv") ,help="labels_file_name")    
    args = parser.parse_args()
    print("args=", args)

     ########## load lables ########
    labels_df=pd.read_csv(f"{args.labels_path}/{args.labels_file_name}")
    labels_dict=dict(zip(labels_df["domain"].tolist(),labels_df["weak_label"].tolist()))
    labels_annot_df=pd.read_csv(f"{args.labels_path}/labels_annot.csv")
    mode="scores_ready"
    if mode=="scores_ready":
        # gat_parq_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/data/weaksupervision/gat_inferred_scores_text_features_pc1_dec2024_from_binary_test.parquet"        
        gat_parq_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/data/weaksupervision/gat_inferred_scores_rni_pc1_dec2024_from_binary_test.parquet"
        reg_dict=search_parquet_duckdb(gat_parq_path, col=None,q_domains=None,max_memory="8GB",schema={'key':'domain','val':'scores'})
        reg_dict={".".join(k.split('.')[::-1]):v for k,v in reg_dict.items()}
        emb_file_name=gat_parq_path.replace("_pc1_dec2024_from_binary_test.parquet","TL_test_set_emb_dict.pkl")
        labels_annot_df=labels_annot_df[labels_annot_df["domain"].isin(reg_dict.keys())]
        labels_annot_df["binary_label"]=labels_annot_df["domain"].apply(lambda x: labels_dict[x])
    else:
        ############## Load model ###############       
        model_file_name = f"{args.model_path}/{args.model_file_name}"
        mlp_reg=None
        with open(model_file_name, 'rb') as f:
            mlp_reg = pickle.load(f) 
        ############## Load emb ###############       
        emb_file_name = f"{args.emb_path}/{args.emb_file_name}"    
        emb_dict={}
        with open(emb_file_name, 'rb') as f:
            emb_dict = pickle.load(f)            
        labels_annot_df=labels_annot_df[labels_annot_df["domain"].isin(emb_dict.keys())]
        labels_annot_df["binary_label"]=labels_annot_df["domain"].apply(lambda x: labels_dict[x])
        ############## Load emb index ###############
        reg_dict={}
        for k,v in tqdm(emb_dict.items()):
            reg_dict[k]=mlp_reg.predict([v])[0]

    labels_annot_df["pred_pc1_score"]=labels_annot_df["domain"].apply(lambda x: reg_dict[x])
    for th in [0.1,0.3,0.5,0.7,0.9]:
        labels_annot_df[f"pred_label_gte_{int(th*10)}"]=labels_annot_df["pred_pc1_score"].apply(lambda x: 1 if x>=th else 0)
    write_results(labels_annot_df,emb_file_name)
    labels_annot_df.to_csv(f"{emb_file_name.replace("_test_set_emb_dict.pkl","_pred_pc1_scores.csv")}",index=None)
    labels_annot_df.to_parquet(f"{emb_file_name.replace("_test_set_emb_dict.pkl","_pred_pc1_scores.parquet")}",
                engine='pyarrow',row_group_size=1000,use_dictionary=True,compression="snappy")
    ##################### ACC Stats ###################
    


if __name__ == '__main__':
    main()