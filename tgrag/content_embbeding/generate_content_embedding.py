import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import pickle
import argparse
def get_ds_files(repo_id,pattern="*.parquet"):
    files = list_repo_files(
        repo_id=repo_id,
        repo_type="dataset")
    files_lst=[]
    for f in files:
        if f.endswith(pattern[1:]):
            files_lst.append(f)
    return files_lst

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Content Embedding")
    parser.add_argument("--hf_repo_id", type=str, default="Hussein-Abdallah/CrediBench-WebContent-Dec2024", help="dqr dataset path")
    parser.add_argument("--emb_model", type=str, default="google/embeddinggemma-300m", help="LLM model Name",
                        choices=["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-4B",
                                 "google/embeddinggemma-300m"])
        parser.add_argument("--local_dir", type=str, default="/home/mila/a/abdallah/hsh_projects/content_emb/content_emb", help="document key column")
    parser.add_argument("--hf_files_start_idx", type=int, default=0, help="start file idx")
    parser.add_argument("--hf_files_end_idx", type=int, default=100, help="end file idx")
    parser.add_argument("--parquet_batch_size", type=int, default=100000, help="parquet file read rows batch_size")
    parser.add_argument("--emb_batch_size", type=int, default=5000, help="GPU emb batch size")
    parser.add_argument("--emb_dim", type=int, default=256, help="resize embedding dim")
    args = parser.parse_args()
    print(f"args={args}")
    ds_name=args.hf_repo_id.split("/")[-1]
    repo_parquet_files_lst = get_ds_files(args.hf_repo_id)
    model = SentenceTransformer(args.emb_model)
    for f_idx, f in tqdm(enumerate(repo_parquet_files_lst[args.hf_files_start_idx:args.hf_files_end_idx+1])):
        hf_hub_download(repo_id=args.hf_repo_id, filename=f, repo_type="dataset", local_dir=f"{args.local_dir}/{ds_name}/")
        # print(f"{local_dir}/{ds_name}/{f}")
        parquet_file = pq.ParquetFile(f"{args.local_dir}/{ds_name}/{f}")
        print(f"file={f}")
        for i, batch in tqdm(enumerate(parquet_file.iter_batches(batch_size=args.parquet_batch_size)),
                             total=1 + int(parquet_file.metadata.num_rows / args.parquet_batch_size)):
            df_chunk = batch.to_pandas()
            urls_lst = df_chunk["WARC_Target_URI"].tolist()
            domains_lst = df_chunk["Domain_Name"].tolist()
            domains_txt = df_chunk["wet_record_txt"].astype(str).tolist()
            embds = {}
            for emb_batch_idx in tqdm(range(0, len(domains_lst), args.emb_batch_size)):
                batch_emb = model.encode(domains_txt[emb_batch_idx:emb_batch_idx + args.emb_batch_size])
                for emb_idx in range(len(batch_emb)):
                    # print("domain_idx=",emb_batch_idx+emb_idx)
                    if domains_lst[emb_batch_idx + emb_idx] not in embds:
                        embds[domains_lst[emb_batch_idx + emb_idx]] = []
                    embds[domains_lst[emb_batch_idx + emb_idx]].append(
                        [urls_lst[emb_batch_idx + emb_idx], batch_emb[emb_idx][0:args.emb_dim]])
            emb_file_name = f'{args.local_dir}/{f.split(".")[0]}_{f_idx}.pkl'
            with open(emb_file_name, 'wb') as emb_file:
                pickle.dump(embds, emb_file)
