import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def test_score_sum(new_nodes: pd.DataFrame, tol: float = 1e-3) -> bool:
    total = new_nodes['importance'].sum()
    logging.info(f'Total importance score: {total}')
    logging.info(f'Expected: 1.0, Difference: {abs(total - 1.0)}')
    return abs(total - 1.0) < tol


def show_score_distribution(new_nodes: pd.DataFrame) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Log scale histogram
    ax1.hist(new_nodes['importance'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Importance Score')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_title('Distribution of Importance Scores (Log Scale)')

    # 2. Top percentile only
    top_percentile = new_nodes['importance'].quantile(0.95)
    top_scores = new_nodes[new_nodes['importance'] >= top_percentile]['importance']
    ax2.hist(top_scores, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Importance Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Top 5% Importance Scores (>= {top_percentile:.6f})')

    # 3. Box plot to show quartiles
    ax3.boxplot(new_nodes['importance'], vert=True)
    ax3.set_ylabel('Importance Score')
    ax3.set_title('Box Plot of Importance Scores')
    ax3.set_xticklabels(['All Nodes'])

    # 4. Top 20 nodes
    top_20 = new_nodes.nlargest(20, 'importance')
    ax4.bar(range(len(top_20)), top_20['importance'], alpha=0.7, color='red')
    ax4.set_xlabel('Node Rank')
    ax4.set_ylabel('Importance Score')
    ax4.set_title('Top 20 Nodes by Importance')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    logging.info(f'\nPageRank Statistics:')
    logging.info(f'Min score: {new_nodes["importance"].min():.8f}')
    logging.info(f'Max score: {new_nodes["importance"].max():.8f}')
    logging.info(f'Mean score: {new_nodes["importance"].mean():.8f}')
    logging.info(f'Median score: {new_nodes["importance"].median():.8f}')
    logging.info(f'95th percentile: {new_nodes["importance"].quantile(0.95):.8f}')
    logging.info(f'99th percentile: {new_nodes["importance"].quantile(0.99):.8f}')


def preprocess_data(
    nodes_path: str, edges_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info('Starting data preprocessing...')

    # Load raw data
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    logging.info(f'Loaded {len(nodes)} nodes and {len(edges)} edges')

    # Handle duplicate nodes
    duplicate_nodes = nodes[nodes.duplicated(subset=['node_id'], keep=False)]
    if not duplicate_nodes.empty:
        logging.info(
            f'WARNING: Found {len(duplicate_nodes)} duplicate node IDs in {nodes_path}'
        )
        logging.info('Duplicate node IDs:')
        logging.info(duplicate_nodes['node_id'].value_counts().sort_index())
        logging.info('\nFirst few duplicate rows:')
        logging.info(duplicate_nodes.head(10))

        original_count = len(nodes)
        nodes = nodes.drop_duplicates(subset=['node_id'], keep='first')
        logging.info(f'\nRemoved {original_count - len(nodes)} duplicate rows')
        logging.info(f'Unique nodes remaining: {len(nodes)}')
    else:
        logging.info(f'No duplicate node IDs found in {nodes_path}')

    # Validate edge consistency
    nodes_from_df = set(nodes['node_id'])
    nodes_from_edges = set(edges['src'].unique()) | set(edges['dst'].unique())

    missing_in_nodes = nodes_from_edges - nodes_from_df
    if missing_in_nodes:
        logging.info(
            f'WARNING: Found {len(missing_in_nodes)} nodes in edges that are not in nodes DataFrame'
        )
        logging.info(f'First few missing nodes: {list(missing_in_nodes)[:10]}')

        # Add missing nodes to nodes DataFrame
        missing_nodes_df = pd.DataFrame({'node_id': list(missing_in_nodes)})
        nodes = pd.concat([nodes, missing_nodes_df], ignore_index=True)
        logging.info(f'Added {len(missing_in_nodes)} missing nodes to nodes DataFrame')

    # Remove self-loops (optional)
    self_loops = edges[edges['src'] == edges['dst']]
    if not self_loops.empty:
        logging.info(f'Found {len(self_loops)} self-loops, removing them...')
        edges = edges[edges['src'] != edges['dst']]

    # Remove duplicate edges (optional)
    original_edge_count = len(edges)
    edges = edges.drop_duplicates(subset=['src', 'dst'], keep='first')
    if len(edges) < original_edge_count:
        logging.info(f'Removed {original_edge_count - len(edges)} duplicate edges')

    logging.info(f'Preprocessing complete: {len(nodes)} nodes, {len(edges)} edges')
    return nodes, edges


def build_adjacency(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> Tuple[List[int], Dict[int, pd.DataFrame], Dict[int, int], Dict[int, List[int]]]:
    all_node_ids = set(nodes['node_id'])
    node_ids = list(all_node_ids)
    len(node_ids)

    adjacency_df = edges.groupby('src')['dst'].apply(set)
    adjacency = {node: adjacency_df.get(node, set()) for node in node_ids}

    out_degree = {node: len(adjacency[node]) for node in node_ids}

    incoming_df = edges.groupby('dst')['src'].apply(list)
    incoming = {node: incoming_df.get(node, []) for node in node_ids}

    return node_ids, adjacency, out_degree, incoming


def check_convergence(
    iteration: int, prev_importance: pd.Series, importance: pd.Series, tol: float
) -> bool:
    diff = abs(prev_importance - importance).sum()
    if iteration % 10 == 0 or iteration < 10:
        logging.info(
            f'Iteration {iteration + 1}: convergence diff = {diff:.8f} (target: {tol})'
        )
    if abs(prev_importance - importance).sum() < tol:
        logging.info(f'Converged after {iteration + 1} iterations')
        return True
    return False


def test_positive_values(nodes: pd.DataFrame) -> bool:
    min_score = nodes['importance'].min()
    return min_score > 0


def test_degree_correlation(nodes: pd.DataFrame, edges: pd.DataFrame) -> bool:
    in_degree = edges['dst'].value_counts()

    test_df = nodes.copy()
    test_df['in_degree'] = test_df['node_id'].map(in_degree).fillna(0)

    correlation = test_df['importance'].corr(test_df['in_degree'])

    logging.info(f'\n--- Degree-PageRank Correlation Test ---')
    logging.info(f'Correlation between in-degree and PageRank: {correlation:.4f}')

    high_degree_threshold = test_df['in_degree'].quantile(0.9)
    low_degree_threshold = test_df['in_degree'].quantile(0.1)

    high_degree_avg = test_df[test_df['in_degree'] >= high_degree_threshold][
        'importance'
    ].mean()
    low_degree_avg = test_df[test_df['in_degree'] <= low_degree_threshold][
        'importance'
    ].mean()

    logging.info(
        f'Avg PageRank for high in-degree nodes (top 10%): {high_degree_avg:.8f}'
    )
    logging.info(
        f'Avg PageRank for low in-degree nodes (bottom 10%): {low_degree_avg:.8f}'
    )

    if len(test_df[test_df['in_degree'] <= low_degree_threshold]) > 0:
        ratio = high_degree_avg / max(low_degree_avg, 1e-10)  # Avoid division by zero
        logging.info(f'Ratio (high/low): {ratio:.2f}')

    plt.figure(figsize=(7, 5))
    plt.scatter(test_df['in_degree'], test_df['importance'], alpha=0.5, s=10)
    plt.xlabel('In-degree')
    plt.ylabel('PageRank Score')
    plt.title('In-degree vs PageRank Correlation')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    if correlation > 0.1 and high_degree_avg > low_degree_avg:
        logging.info('Positive correlation between degree and PageRank')
        return True
    else:
        logging.info('Warning: Weak correlation between degree and PageRank')
        return False
