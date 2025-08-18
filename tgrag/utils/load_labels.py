import os
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from pandas import DataFrame

from tgrag.utils.matching import extract_graph_domains
from tgrag.utils.path import get_root_dir


def load_credibility_scores(path: str, use_core: bool = False) -> pd.DataFrame:
    cred_df = pd.read_csv(path)
    return cred_df[['domain', 'pc1']]


def get_target_set() -> set[str]:
    """Get a list (set) of target domains."""
    path = os.path.join(get_root_dir(), 'data', 'target_set.txt')
    wanted_domains = set()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                domain = parts[0].strip()
                if domain:  # skip empty lines
                    wanted_domains.add(domain)

    return wanted_domains


def get_labelled_set() -> Set[str]:
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


def get_labelled_dict() -> Dict[str, float]:
    """Get a dict of labelled domains, score."""
    path = os.path.join(get_root_dir(), 'data', 'dqr', 'domain_pc1.csv')
    wanted_domains = {}

    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            wanted_domains[parts[0]] = float(parts[1])

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


def match_labels_to_graph(cred_df: DataFrame, vert_df: DataFrame) -> DataFrame:
    """Merge cred_df (with 'domain', 'pc1') and vert_df (with 'nid', 'domain')
    on 'domain', returning a DataFrame with columns ['nid', 'cred_score'].
    """
    merged = vert_df.merge(cred_df, on='domain', how='inner')
    return merged[['nid', 'pc1']].rename(columns={'pc1': 'cred_score'})


def generate_labels(data_path: str, label_path: Path, slice: str) -> Path:
    cred_scores_path = f'{label_path}/data/dqr/domain_pc1.csv'
    vertices_path = os.path.join(data_path, 'output', 'vertices.1M.txt.gz')
    output_csv_path = os.path.join(data_path, 'output', 'targets.csv')
    vertices_df = extract_graph_domains(vertices_path)

    cred_df = load_credibility_scores(cred_scores_path, vertices_df)

    labels_df = match_labels_to_graph(cred_df, vertices_df)

    labels_df.to_csv(output_csv_path, index=False)

    return Path(output_csv_path)
