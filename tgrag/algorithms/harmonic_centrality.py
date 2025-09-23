import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from tgrag.utils.logger import setup_logging
from tgrag.utils.pagerank_utils import *
from tgrag.utils.pagerank_utils import preprocess_data
from tgrag.utils.path import get_root_dir, get_scratch
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Generate HarmonicCentrality.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--node-file',
    type=str,
    default='data/crawl-data/manual/temporal_nodes.csv',
    help='Path to the feature file in a csv format.',
)
parser.add_argument(
    '--edge-file',
    type=str,
    default='data/crawl-data/manual/temporal_edges.csv',
    help='Path to the edge file in a csv format.',
)
parser.add_argument(
    '--use-scratch',
    action='store_true',
    help='Whether to use scratch location as data root.',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='script_pagerank.log',
    help='Name of log file at project root.',
)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Seed for reproducibility.',
)


def calculate_harmonic_centrality_sampled(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    sample_size: int = 300000,
    directed: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    logging.info(f'Sampling {sample_size:,} nodes from {len(nodes):,} total nodes')
    sampled_node_ids = np.random.choice(
        nodes['node_id'].values, size=sample_size, replace=False
    )
    sampled_nodes = nodes[nodes['node_id'].isin(sampled_node_ids)]
    sampled_edges = edges[
        edges['src'].isin(sampled_node_ids) & edges['dst'].isin(sampled_node_ids)
    ]
    logging.info(
        f'Sampled subgraph: {len(sampled_nodes):,} nodes, {len(sampled_edges):,} edges'
    )
    if directed:
        G = nx.from_pandas_edgelist(
            sampled_edges, source='src', target='dst', create_using=nx.DiGraph()
        )
    else:
        G = nx.from_pandas_edgelist(sampled_edges, source='src', target='dst')
    G.add_nodes_from(sampled_node_ids)
    harmonic_centrality = nx.harmonic_centrality(G)
    sampled_nodes_with_centrality = sampled_nodes.copy()
    sampled_nodes_with_centrality['harmonic_centrality'] = (
        sampled_nodes_with_centrality['node_id'].map(harmonic_centrality)
    )
    return sampled_nodes_with_centrality


def main() -> None:
    args = parser.parse_args()
    if args.use_scratch:
        root = get_scratch()
    else:
        root = get_root_dir()
    setup_logging(args.log_file)
    seed_everything(args.seed)

    root = get_root_dir()

    nodes, edges = preprocess_data(root / args.node_file, root / args.edge_file)
    logging.info(f'Number of nodes: {len(nodes):,}')
    logging.info(f'Number of edges: {len(edges):,}')
    logging.info(f'Graph density: {len(edges) / (len(nodes) * (len(nodes) - 1)):.6f}')
    harmonic_nodes = calculate_harmonic_centrality_sampled(
        nodes, edges, sample_size=300000, directed=True, seed=args.seed
    )

    centrality_values = harmonic_nodes['harmonic_centrality'].dropna()
    nonzero_vals = centrality_values[centrality_values > 0]

    plt.figure(figsize=(12, 8))
    log_vals = np.log10(nonzero_vals + 1e-10)
    sns.kdeplot(log_vals, fill=True, alpha=0.7, color='steelblue', linewidth=2)
    plt.ylabel('Density', fontsize=12)
    plt.title('Harmonic Centrality Distribution (KDE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    xticks = ax.get_xticks()
    actual_values = [f'{10**x:.0f}' if x >= 0 else f'{10**x:.2f}' for x in xticks]
    ax.set_xticklabels(actual_values)
    plt.xlabel('Harmonic Centrality', fontsize=12)
    total_nodes = len(centrality_values)
    zero_nodes = (centrality_values == 0).sum()
    nonzero_nodes = len(nonzero_vals)
    stats_text = f"""
    Total sampled nodes: {total_nodes:,}
    Isolated nodes (HC=0): {zero_nodes:,} ({zero_nodes / total_nodes * 100:.1f}%)
    Connected nodes (HC>0): {nonzero_nodes:,} ({nonzero_nodes / total_nodes * 100:.1f}%)
    Max HC: {centrality_values.max():.1f}
    """
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
