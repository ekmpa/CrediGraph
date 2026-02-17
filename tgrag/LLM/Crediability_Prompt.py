import asyncio
from openai import OpenAI,AsyncOpenAI
from dotenv import load_dotenv
from glob import glob
import os
import json
import pandas as pd
import pickle
from tqdm import tqdm

async def get_completion_pc1(client, prompt):
    completion = await client.chat.completions.create(
              model="gpt-5",
              messages=[
                {"role": "developer", "content": "You are a web domain credibility assessment assistant. Given a web domain and a sample content, assign a credibility score from 0 to 10 where with 10 being most credible and 0 being least credible. "},
                {"role": "user", "content":prompt }
              ]
            )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

async def get_completion_mbfc(client, prompt):
    completion = await client.chat.completions.create(
              model="gpt-5",
              messages=[
                {"role": "developer", "content": "You are a web domain credibility assessment assistant. Given a web domain and a sample content, assign a Media Bias/Fact Check credibility score from 0 to 10 where with 10 being most credible and 0 being least credible. "},
                {"role": "user", "content":prompt }
              ]
            )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

async def main():
    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Replace with your actual key

    # for month in ['oct','nov','dec']:
    target="mbfc"
    domain_ratings_df=pd.read_csv("/shared_mnt/github_repos/CrediGraph/data/dqr/domain_ratings.csv")
    domain_ratings_dict=dict(zip(domain_ratings_df["domain"],domain_ratings_df[target]))
    test_set_domains=pd.read_csv(f"/shared_mnt/github_repos/CrediGraph/plots/dqr_testset_nov_{target}_text_embeddingTE3L.csv")["domain"].tolist()

    for month in ['oct','nov','dec']:
        domains_content_df=pd.read_csv(f"/shared_mnt/github_repos/CrediGraph/data/dqr/dqr_{month}_domains_content.csv")
        # print(domains_content_df.columns)
        print(f"month={month}")
        if "Domain_Name" in domains_content_df.columns:
            domains_content_df.rename(columns={'Domain_Name': 'url'}, inplace=True)            
        domains_content_df=domains_content_df[domains_content_df["url"].isin(test_set_domains)]
        domains_content_df[target]=domains_content_df["url"].apply(lambda x:domain_ratings_dict[x])   
        domains_content_df[f"{target}_rounded"]=round(domains_content_df[target],2)
        domains_content_df=domains_content_df.sort_values(by=[target]) 
        answers_dict={}
        prompts_lst=[]
        for i,row in domains_content_df.iterrows():
            if "title" in domains_content_df.columns:
                domain_content=f"""domain name: {str(row["url"])}
            domain titile: {row["title"]}
            domain description: { row["desc"]}
            domain content: {str(row["text"])[0:10000]}"""
            else:
                domain_content=f"""domain name: {str(row["url"])}
            domain content: {str(row["txt"])[0:10000]}"""
            if target=="pc1":
                message=f"Assign a credibility score from 0 to 10, where 10 is being most credible and 0 is being least credible.\nReturn your answer in format Score=<sscore>.\n\n {domain_content}"
                get_completion=get_completion_pc1
            elif target =="mbfc":
                message=f"Assign a   Media Bias/Fact Check (MBFC) credibility score from 0 to 10, where 10 is being most credible and 0 is being least credible. the possible scores are 0,2,4,6,8,10. \nReturn your answer in format Score=<sscore>.\n\n {domain_content}"
                get_completion=get_completion_mbfc

           
            prompts_lst.append([row["url"],row[target],message])
        batch_size=50
        answers_dict={}
        for i in tqdm(range(0,len(prompts_lst),10)):
            # print("i=",i)
            tasks = [get_completion(client, prompt[2]) for prompt in prompts_lst[i:i+batch_size]]
            results = await asyncio.gather(*tasks)
            # print("results=",results)
            for bs_i in range(len(results)):
                answers_dict[prompts_lst[i+bs_i][0]]=[prompts_lst[i+bs_i][1],prompts_lst[i+bs_i][2],results[bs_i]]
                
        file_name=f'dqr_{month}_{target}_gpt5_chat_results.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(answers_dict, f) 

if __name__ == "__main__":
    asyncio.run(main())