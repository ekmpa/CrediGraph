import argparse
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
import string
nltk.download('stopwords')
stop_words = stopwords.words('english')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
def clean_text(text):
    '''
    Perform stop-words removal and lemmatization
    '''
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+|http\S+", " ", text)
    text = ''.join([char for char in text if char not in string.punctuation])
    words = [word for word in text.split() if word not in stopwords.words('english')]
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return " ".join(words)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Embedding")
    parser.add_argument("--data_path", type=str, default="../../data/dqr/", help="dqr dataset path")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"], help="CrediBench month snapshot")
    args = parser.parse_args()

    dqr_df = pd.read_csv(f"{args.data_path}dqr_{args.month}_domains_content.csv")
    dqr_df.columns = ["html", "url"]
    df_phishtank = pd.read_csv(f"{args.data_path}cc_{args.month}_2024_phishtank_domains_scraped_metadata_0.csv")
    df_phishtank = df_phishtank[["html", "url"]]
    df_URLhaus = pd.read_csv(f"{args.data_path}cc_{args.month}_2024_URLhaus_domains_scraped_metadata_0.csv")
    df_URLhaus = df_URLhaus[["html", "url"]]
    df_phishDataset_legit = pd.read_csv(f"{args.data_path}cc_{args.month}_2024_PhishDataset_legit_domains_scraped_metadata_0.csv")
    df_phishDataset_legit = df_phishDataset_legit[["html", "url"]]

    df = pd.concat([dqr_df, df_phishtank, df_URLhaus, df_phishDataset_legit])
    dqr_df[dqr_df["html"].isna()]
    df["final_text"] = df["html"]
    ## Clean Text
    df['final_text_cleaned'] = df['final_text'].apply(clean_text)
    ## TFIDF
    tfidf_vectorizer = TfidfVectorizer(min_df=150, ngram_range=(1, 2))
    tfidf = tfidf_vectorizer.fit_transform(df['final_text_cleaned'])
    ## Write TFIDF emb
    tfidf_emb_dict = {}
    for i in range(0, tfidf.shape[0]):
        tfidf_emb_dict[df.iloc[i]["url"]] = tfidf[i].toarray()[0]

    start_idx=0
    end_idx=len(dqr_df)
    with open(f'{args.data_path}dqr_oct_TFIDF_weaksupervision_emb_{tfidf.shape[1]}.pkl', 'wb') as f:
        pickle.dump(dict(list(tfidf_emb_dict.items())[start_idx:end_idx]), f)
    start_idx = end_idx
    end_idx+=len(df_phishtank)
    with open(f'{args.data_path}phishtank_oct_TFIDF_weaksupervision_emb_{tfidf.shape[1]}.pkl', 'wb') as f:
        pickle.dump(dict(list(tfidf_emb_dict.items())[start_idx:end_idx]), f)
    start_idx = end_idx
    end_idx +=  len(df_URLhaus)
    with open(f'{args.data_path}URLhaus_oct_TFIDF_weaksupervision_emb_{tfidf.shape[1]}.pkl', 'wb') as f:
        pickle.dump(dict(list(tfidf_emb_dict.items())[start_idx:end_idx]), f)
    start_idx = end_idx
    end_idx += len(df_phishDataset_legit)
    with open(f'{args.data_path}phishDataset_legit_oct_TFIDF_weaksupervision_emb_{tfidf.shape[1]}.pkl', 'wb') as f:
        pickle.dump(dict(list(tfidf_emb_dict.items())[start_idx:end_idx]), f)




