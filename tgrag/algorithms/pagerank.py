import pandas as pd
import matplotlib.pyplot as plt

#   Placeholder file for pagerank algorithm implementation.

TEMPORAL_EDGES_PATH = "../../data/dqr/temporal_edges.csv"
TEMPORAL_NODES_PATH = "../../data/dqr/temporal_nodes.csv"

nodes = pd.read_csv(TEMPORAL_NODES_PATH)
edges = pd.read_csv(TEMPORAL_EDGES_PATH)

def pagerank(nodes, edges, damping=0.85, max_iter=100, tol=1e-6):
    def test_convergence(prev, curr):
        return abs(prev - curr).sum() < tol

    node_ids = nodes['node_id'].tolist()
    N = len(node_ids)
    importance = pd.Series(1.0 / N, index=node_ids)

    # Build adjacency list using correct column names
    adjacency = {node: set() for node in node_ids}
    for _, row in edges.iterrows():
        adjacency[row['src']].add(row['dst'])

    for _ in range(max_iter):
        prev_importance = importance.copy()
        for node in node_ids:
            incoming = [src for src, targets in adjacency.items() if node in targets]
            rank_sum = sum(prev_importance[src] / len(adjacency[src]) if adjacency[src] else 0 for src in incoming)
            importance[node] = (1 - damping) / N + damping * rank_sum

        if test_convergence(prev_importance.values, importance.values):
            break

    new_nodes = nodes.copy()
    new_nodes['importance'] = new_nodes['node_id'].map(importance)
    return new_nodes

new_nodes = pagerank(nodes, edges)
print(test_score_sum(new_nodes))
show_score_distribution(new_nodes)

def test_score_sum(new_nodes, tol=1e-6):
    total = new_nodes['importance'].sum()
    return abs(total - 1.0) < tol

def show_score_distribution(new_nodes):
    plt.figure(figsize=(8, 5))
    plt.hist(new_nodes['importance'], bins=10, edgecolor='black')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Importance Scores')
    plt.show()