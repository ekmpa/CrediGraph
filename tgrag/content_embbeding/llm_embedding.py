from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import pickle
import ollama
import argparse
def embd_documents(model_name,documents_path,k_col,text_col,batch_size=100, useOllama=False):
    data_df=pd.read_csv(documents_path)
    model = SentenceTransformer(model_name)
    keys_lst = data_df[k_col].tolist()
    docs_lst = data_df[text_col].astype(str).tolist()
    embds = {}
    for emb_batch_idx in range(0, len(keys_lst), batch_size):
        # print("emb_batch_idx=", emb_batch_idx)
        if useOllama:
            response = ollama.embed(model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
                                    input=docs_lst[emb_batch_idx:emb_batch_idx + batch_size])
            batch_emb = response["embeddings"]
        else:
            batch_emb=model.encode(docs_lst[emb_batch_idx:emb_batch_idx+batch_size])
        for emb_idx in range(len(batch_emb)):
            embds[keys_lst[emb_batch_idx + emb_idx]] = batch_emb[emb_idx]
    return embds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Embedding")
    parser.add_argument("--documents_path", type=str, default="../../data/dqr/domain_pc1.csv", help="dqr dataset path")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="LLM model Name", choices=["Qwen/Qwen3-Embedding-0.6B","Qwen/Qwen3-Embedding-8B","Qwen/Qwen3-Embedding-4B","google/embeddinggemma-300m"])
    parser.add_argument("--k_col", type=str, default="domain", help="document key column")
    parser.add_argument("--text_col", type=str, default="text", help="document text column")
    parser.add_argument("--batch_size", type=int, default=100, help="scarp batch_size")
    args = parser.parse_args()
    embddings=embd_documents(args.model_name,args.documents_path,args.k_col,args.text_col,args.batch_size)
    # with open(f'dqr_{emb_model.split("/")[-1]}_{embds['sports.nbcsports.com'].shape[0]}.pkl', 'wb') as f:
    with open(f'/shared_mnt/dqr_{args.model_name.split("/")[-1]}_{len(embddings[list(embddings.keys())[0]])}.pkl', 'wb') as f:
        pickle.dump(embddings, f)