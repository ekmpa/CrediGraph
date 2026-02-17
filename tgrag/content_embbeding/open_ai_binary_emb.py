import asyncio
from openai import OpenAI,AsyncOpenAI
from dotenv import load_dotenv
from glob import glob
import os
import json
import pandas as pd
import pickle
from tqdm import tqdm
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import experiments
from experiments.mlp_experiments.utils import search_parquet_duckdb
import pickle
dotenv_path="/home/mila/a/abdallah/scratch/hsh_projects/config_keys/.env"
data_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/data"
out_path="/home/mila/a/abdallah/scratch/hsh_projects/CrediGraph/data/scrapedContent/weaklabels/TE3L"
text_max_length=4500
def open_ai_emb(client,docs_lst,domains_lst,bs=100,out_batch_prefix="binary_TE3L_domains_woc_emb_",model_name="text-embedding-3-large"):
    embd_dict={}
    for batch_idx,doc_idx in enumerate(tqdm(range(0,len(docs_lst),bs))):
        response = client.embeddings.create(
                model=model_name,
                input=docs_lst[doc_idx:doc_idx+bs])
        for url_idx,emb in enumerate(response.data):
            embd_dict[domains_lst[doc_idx+url_idx]]=emb.embedding

        if (batch_idx+1)%100==0 or doc_idx+bs>=len(docs_lst):    
            file_name=f'{out_path}/{out_batch_prefix}_{len(embd_dict[list(embd_dict.keys())[0]])}_batch_{batch_idx}.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(embd_dict, f)  
            del embd_dict
            embd_dict={}
    
def emb_doamins_wc(client):
    domain_wc_df=pd.read_csv(f"{data_path}/scrapedContent/weaklabels/weaklabel_domains_with_content.csv")
    docs_lst=["Web Domain Name:"+elem[0:text_max_length] for elem in  domain_wc_df["text"].to_list()]
    domains_lst=domain_wc_df["domain"]
    open_ai_emb(client,docs_lst,domains_lst,bs=100,out_batch_prefix="binary_TE3L_domains_wc_emb")
def emb_doamins_woc(client):
    domain_woc_df=pd.read_csv(f"{data_path}/scrapedContent/weaklabels/weaklabel_domains_without_content.csv")
    docs_lst=["Web Domain Name:"+elem[0:text_max_length] for elem in  domain_woc_df["domain"].to_list()]
    domains_lst=domain_woc_df["domain"]
    open_ai_emb(client,docs_lst,domains_lst,bs=100,out_batch_prefix="binary_TE3L_domains_woc_emb")
def emb_month_doamins_woc(client):
    for month in ["Oct","Nov","Dec"]:
        print(f"########## Emb {month} Month Content ###########")
        file_path=f"{data_path}/weaksupervision/weaksupervision_content_{month}2024.parquet"
        month_content_df=search_parquet_duckdb(file_path, col=None,q_domains=None,max_memory="8GB",schema=None)
        docs_lst=["Web Domain Name:"+month_content_df["pages"].tolist()[0][0]['wet_record_txt'][0:text_max_length] for elem in  month_content_df["pages"].tolist()]        
        domains_lst=month_content_df["domain"]
        del month_content_df
        open_ai_emb(client,docs_lst,domains_lst,bs=100,out_batch_prefix=f"binary_TE3L_domains_{month}_emb")

def main():
    load_dotenv(dotenv_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Replace with your actual key           
    # emb_doamins_wc(client)
    # emb_doamins_woc(client)
    emb_month_doamins_woc(client)  
if __name__ == "__main__":
    main()