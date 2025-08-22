import pandas as pd
import os
from tqdm import tqdm
import threading
import cloudscraper
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
import argparse

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
results = []
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Content-Type': 'text/html; charset=UTF-8',
    "Accept-Language": "en-US,en;q=0.9",
    }
results = []
scraper = cloudscraper.create_scraper()
def scrape_multithread(url):
    global results
    results.append(scrape(url))
def scrape(url):
    # start = time.time()
    scraper = cloudscraper.create_scraper()
    # print(f"############URL({url})###################")
    try:
        response = scraper.get("https://www." + url, headers=headers, timeout=5)
    except:
        try:
            response = scraper.get("https://" + url, headers=headers, timeout=5)
        except:
            try:
                response = scraper.get("http://www." + url, headers=headers, timeout=5)
            except:
                try:
                    response = scraper.get("http://" + url, headers=headers, timeout=5)
                except Exception as e:
                    return [None, str(e), None, None, url]
    soup = BeautifulSoup(response.text, 'html.parser')
    res = []
    try:
        title = soup.find('title').text if soup.find('title') else 'No title found'
        res.append(title)
    except:
        res.append(None)
    try:
        description_meta = soup.find('meta', attrs={'name': 'description'})
        description = description_meta['content'] if description_meta else 'No description found'
        res.append(description)
        # print(f"Description: {description}")
    except:
        res.append(None)

    # Get Open Graph title
    try:
        og_title_meta = soup.find('meta', attrs={'property': 'og:title'})
        og_title = og_title_meta['content'] if og_title_meta else 'No OG title found'
        res.append(og_title)
    except:
        res.append(None)
    res.append(soup.get_text())
    res.append(url)
    scraper.close()
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSite Home Page Scraper")
    parser.add_argument("--domains_path", type=str, default="../../data/dqr/domain_pc1.csv")
    parser.add_argument("--output_path", type=str, default="../../data/scrape/")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    if not os.path.exists(f"../../data/scrapedContent/"):
        os.mkdir(f"../../data/scrapedContent/")

    domains_df = pd.read_csv(args.domains_path)
    domains_lst = domains_df["domain"].tolist()
    del domains_df
    print(f"len of domains={len(domains_lst)}")
    batch_size = args.batch_size
    for b_idx in tqdm(range(0, len(domains_lst) + batch_size, batch_size)):
        threads = []
        results = []
        ############### MultiThreading ###########
        for url in domains_lst[b_idx:b_idx + batch_size]:
            thread = threading.Thread(target=scrape_multithread, args=(url,))
            threads.append(thread)
        for i in tqdm(range(0, len(threads), 1000)):
            for thread in threads[i:i + 1000]:
                thread.start()
            for thread in threads[i:i + 1000]:
                thread.join()
        results_df = pd.DataFrame(results, columns=["title", "desc", "OG-Title", "text", "url"])
        results_df.to_csv(f"../../data/scrapedContent/{args.domains_path.split('/')[-1].split('.')[0]}_scraped_homepage_{b_idx}.csv", index=None, encoding='utf-8',errors='ignore')
