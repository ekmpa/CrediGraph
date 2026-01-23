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

# shard_domains_dict={}
# def process_chunk(chunk):
#     global shard_domains_dict
#     for k, v in chunk:
#         shard_domains_dict[v].append(k)
text_emb_shards_dict={}
domain_index_text_dict, domain_index_set={},()
domain_index_gnn_dict, index_domain_gnn_dict={},{}
def load_emb_index(path="../../../data/Dec2024/gnn_random_v0", index_pickle="domain_shard_index.pkl",invert=True ):
    with open(f'{path}/{index_pickle}', 'rb') as f:
            index_dict = pickle.load(f)
    # global shard_domains_dict
    shard_domains_dict=None
    if invert:
        shard_domains_dict={}
        val_set=set(index_dict.values())
        for v in val_set:
            shard_domains_dict[v]=[]

        for k,v in index_dict.items():
            shard_domains_dict[v].append(k)
    return index_dict,shard_domains_dict

def load_emb_dict(path="../../../data/Dec2024/gnn_random_v0", emb_pickle="domain_shard_index.pkl"):
    shared_key=emb_pickle.split('.')[0]
    if shared_key not in text_emb_shards_dict:        
        with open(f'{path}/{emb_pickle}', 'rb') as f:
            embd_dict = pickle.load(f)
        if isinstance(embd_dict[list(embd_dict.keys())[0]], list):
            embd_dict={ k:v[0][1] for k,v in embd_dict.items()}
        text_emb_shards_dict[shared_key]=embd_dict
    return text_emb_shards_dict[shared_key]

def search_parquet_gnn(path="../../../data/Dec2024/gnn_random_v0", parquet_name="shard_0.parquet", column_name="",search_values=[]):
    column_name = 'domain'
    filters = [(column_name, 'in', search_values)]
    table = pq.read_table(f'{path}/{parquet_name}', filters=filters)
    emb_dict={}
    for batch in table.to_batches():
        for i in range(len(batch)):
            emb_dict[str(batch['domain'][i])]=batch['emb'][i].as_py()
    return emb_dict
def search_parquet_content(path="../../../data/Dec2024/gnn_random_v0", parquet_name="shard_0.parquet", column_name="",search_values=[]):
    column_name = 'domain'
    filters = [(column_name, 'in', search_values)]
    table = pq.read_table(f'{path}/{parquet_name}', filters=filters)
    emb_dict={}
    for batch in table.to_batches():
        for i in range(len(batch)):
            emb_dict[str(batch['domain'][i])]=batch['embeddings'][i][0]['emb'].as_py() ## first doc emb
    return emb_dict

def process_inference_batch(idx,mlp_reg,batch_domains,gnn_emb_path,text_emb_path,gnn_emb_idx):
    # batch_domains = gnn_v[i:i + batch_size]
    print(f"started: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}")    
    global index_domain_text_dict,domain_index_set
    gnn_emb_dict=search_parquet_gnn(gnn_emb_path, parquet_name=f"shard_{gnn_emb_idx}.parquet", column_name="domain",search_values=batch_domains)
    text_emb_index_batch_dict={}
    text_emb_index_batch_dict[-1]=[] ## domains without text content emb
    for k in domain_index_set:
        text_emb_index_batch_dict[k]=[] 

    for k in gnn_emb_dict:
        if k not in domain_index_text_dict:                    
                domain_index_text_dict[k]=-1
        text_emb_index_batch_dict[domain_index_text_dict[k]].append(k)
    
    batch_scores_dict={}
    for text_emb_shard,domains in tqdm(text_emb_index_batch_dict.items()):
        if text_emb_shard==-1:
            text_emb_dict={domain:[0]*256 for domain in domains}
        elif len(domains)>0:
            text_emb_dict=search_parquet_content(text_emb_path, parquet_name=f"{text_emb_shard}.parquet", column_name="domain",search_values=domains)            
        else:
            continue     
        for domain in domains:
            gnn_emb_dict[domain].extend(text_emb_dict[domain])
            
    final_emb_dict={}
    final_emb_dict=gnn_emb_dict        
    if len(final_emb_dict)>0:
        pred_scores=mlp_reg.predict(list(final_emb_dict.values()))
        socres_dict=dict(zip(list(final_emb_dict.keys()), list(pred_scores)))
        batch_scores_dict.update(socres_dict)
            # print(f"finished: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}\ttext_emb_shard{text_emb_shard}")    
    print(f"finished: gnn_emb_idx={gnn_emb_idx}\t batch_idx={idx}")
    return (batch_scores_dict,idx)
def result_callback(result):
    print(f"\nCallback result received for batch idx={result[1]}\n")

def main() -> None:
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Inference")
    parser.add_argument("--text_emb_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/content_emb/content_emb/nov2024_wetcontent_embeddinggemma-300m/Credibench_nov2024_wetcontent_embeddinggemma-300m/") ,help="text emb files path")
    parser.add_argument("--gnn_emb_path", type=str, default=str(root + "/data/Nov2024/gnn_random/") ,help="gnn emb files path")
    parser.add_argument("--model_path", type=str, default=str("/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/plots/"),help="model path")
    parser.add_argument("--month", type=str, default="nov", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--exec_mode", type=str, default="multiprocess", choices=["serial", "multithread", "multiprocess"],help="execution mode")
    args = parser.parse_args()
    print("args=", args)
    ############## Load Model ###############       
    modelname = f"mlp_model_{args.month}_pc1_text_embeddinggemma-300m_GNN-RNI.pkl"
    mlp_reg=None
    with open(f'{args.model_path}/{modelname}', 'rb') as f:
        mlp_reg = pickle.load(f)    
    X_test_feat = [[0]*256*2]
    pred = mlp_reg.predict(X_test_feat)
    ############## Load emb index ###############
    global domain_index_text_dict, domain_index_set   
    global domain_index_gnn_dict, index_domain_gnn_dict
    domain_index_text_dict, domain_index_set    = load_emb_index(args.text_emb_path, f"{args.month}2024_wetcontent_domains_index.pkl",False)
    domain_index_set=set(domain_index_text_dict.values())
    domain_index_gnn_dict, index_domain_gnn_dict = load_emb_index(args.gnn_emb_path, "domain_shard_index.pkl",True)
    batch_size=int(4e5)
    print(f'batch_size={batch_size}')
    print(f'cpu_count={cpu_count()}')
    for gnn_k,gnn_v in tqdm(index_domain_gnn_dict.items()):
        if int(gnn_k) not in [3,6,9]:
            print(f"Skipping shard:{gnn_k}")
            continue            
        ############### serial Execution ###############
        print(f"shared {gnn_k} len of domains={len(gnn_v)}")
        if args.exec_mode=="serial":
            results=[]
            for i in tqdm(range(0, len(gnn_v), batch_size)):
                results.append(process_inference_batch(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k))
        ############### parallel Execution ###############            
        elif args.exec_mode == "multiprocess":
            data=[(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k) for i in range(0, len(gnn_v), batch_size)]       
            print(f"Total number of batches={len(data)}")
            ########## Multiprocessing Pool ###############        
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.starmap(process_inference_batch, data)
            # print(results)
            pool.close()
        ########### Multithreading ###############
        elif args.exec_mode == "multithread":
            data=[(i,mlp_reg,gnn_v[i:i + batch_size],args.gnn_emb_path,args.text_emb_path,gnn_k) for i in range(0, len(gnn_v), batch_size)]       
            threads = []
            for d in data:
                threads.append(threading.Thread(target=process_inference_batch, args=d))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        ########### Save Results###############
        batch_infer_dict={}
        for res in results:           
            batch_infer_dict.update(res[0])
        results=None
        pd.DataFrame(list(zip(batch_infer_dict.keys(), batch_infer_dict.values())),columns=["domain", "pc1_score"])\
        .to_parquet(f"{args.model_path}/mlpInfer_{args.month}2024_pc1_embeddinggemma-300m_GNN-RNI_shard_{gnn_k}.parquet",
                    engine='pyarrow',row_group_size=50_000,use_dictionary=True,compression="snappy")

if __name__ == '__main__':
    main()