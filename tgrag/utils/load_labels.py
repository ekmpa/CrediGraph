import os
from pathlib import Path

import pandas as pd

from tgrag.utils.matching import extract_graph_domains
from tgrag.utils.path import get_root_dir

def load_credibility_scores(path: str, use_core: bool = False) -> pd.DataFrame:
    cred_df = pd.read_csv(path)
    cred_df['match_domain'] = cred_df['domain']
    return cred_df[['match_domain', 'pc1']]


def get_labelled_set() -> set[str]:
    """Get a list (set) of labelled domains."""
    path = os.path.join(get_root_dir(), 'data', 'dqr', 'domain_pc1.csv')
    wanted_domains = set()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                domain = parts[0].strip()
                if domain:  # skip empty lines
                    wanted_domains.add(domain)

    # print(f'[INFO] Found {len(wanted_domains)} domains ')
    return wanted_domains


def get_credibility_intersection(
    data_path: str, label_path: Path, time_slice: str
) -> None:
    cred_scores_path = f'{label_path}/data/dqr/domain_pc1.csv'
    vertices_path = os.path.join(data_path, 'output_text_dir', 'vertices.txt.gz')
    output_csv_path = os.path.join(data_path, 'output_text_dir', 'vertices.csv')

    print(f'Opening vertices file: {vertices_path}')

    cred_df = load_credibility_scores(cred_scores_path)
    vertices_df = extract_graph_domains(vertices_path)

    enriched_df = pd.merge(vertices_df, cred_df, on='match_domain', how='left')
    enriched_df['pc1'] = enriched_df['pc1'].fillna(-1)
    enriched_df = enriched_df[
        enriched_df['match_domain'].notnull() & (enriched_df['match_domain'] != '')
    ]  # drop empty domains
    enriched_df.to_csv(output_csv_path, index=False)

    print(f'INFO: Merge done. Annotated file saved to {output_csv_path}')

    # After loading
    graph_domains_set = set(vertices_df['match_domain'].unique())
    cred_labels_set = set(cred_df['match_domain'].unique())

    # Node annotation stats
    annotated_nodes = (enriched_df['pc1'] != -1).sum()
    total_nodes = len(vertices_df)
    node_percentage = (annotated_nodes / total_nodes) * 100

    # Label coverage stats (truth labels matched at least once)
    matched_labels = len(cred_labels_set.intersection(graph_domains_set))
    total_labels = len(cred_labels_set)
    label_percentage = (matched_labels / total_labels) * 100

    print(
        f'{annotated_nodes} / {total_nodes} nodes annotated with credibility scores ({node_percentage:.2f}%).'
    )
    print(
        f'{matched_labels} / {total_labels} credibility labels matched at least once on the graph ({label_percentage:.2f}%).'
    )
