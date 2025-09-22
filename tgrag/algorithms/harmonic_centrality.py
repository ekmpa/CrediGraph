import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tgrag.utils.pagerank_utils import *
from tgrag.utils.path import get_root_dir, get_scratch

SEED = 42
np.random.seed(SEED)

node_file = "data/dqr/vertices.csv"
edge_file = "data/dqr/edges.csv"
root = get_root_dir()

nodes, edges = preprocess_data(root / node_file, root / edge_file)
print(f"Number of nodes: {len(nodes):,}")
print(f"Number of edges: {len(edges):,}")
print(f"Graph density: {len(edges) / (len(nodes) * (len(nodes) - 1)):.6f}")

def calculate_harmonic_centrality_sampled(nodes: pd.DataFrame, edges: pd.DataFrame, 
                                        sample_size=300000, directed=True, seed=SEED):
    print(f"Sampling {sample_size:,} nodes from {len(nodes):,} total nodes")
    sampled_node_ids = np.random.choice(nodes['node_id'].values, size=sample_size, replace=False)
    sampled_nodes = nodes[nodes['node_id'].isin(sampled_node_ids)]
    sampled_edges = edges[
        edges['src'].isin(sampled_node_ids) & 
        edges['dst'].isin(sampled_node_ids)
    ]
    print(f"Sampled subgraph: {len(sampled_nodes):,} nodes, {len(sampled_edges):,} edges")
    if directed:
        G = nx.from_pandas_edgelist(sampled_edges, source='src', target='dst', create_using=nx.DiGraph())
    else:
        G = nx.from_pandas_edgelist(sampled_edges, source='src', target='dst')
    G.add_nodes_from(sampled_node_ids)
    harmonic_centrality = nx.harmonic_centrality(G)
    sampled_nodes_with_centrality = sampled_nodes.copy()
    sampled_nodes_with_centrality['harmonic_centrality'] = sampled_nodes_with_centrality['node_id'].map(harmonic_centrality)
    return sampled_nodes_with_centrality

harmonic_nodes = calculate_harmonic_centrality_sampled(nodes, edges, sample_size=300000, directed=True, seed=SEED)

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
Isolated nodes (HC=0): {zero_nodes:,} ({zero_nodes/total_nodes*100:.1f}%)
Connected nodes (HC>0): {nonzero_nodes:,} ({nonzero_nodes/total_nodes*100:.1f}%)
Max HC: {centrality_values.max():.1f}
"""
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()
