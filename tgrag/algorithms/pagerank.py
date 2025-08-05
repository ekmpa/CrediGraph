import pandas as pd
import matplotlib.pyplot as plt

TEMPORAL_EDGES_PATH = "../../data/dqr/temporal_edges.csv"
TEMPORAL_NODES_PATH = "../../data/dqr/temporal_nodes.csv"

NEW_TEMPORAL_EDGES_PATH = "../../data/dqr/new_temporal_edges.csv"
NEW_TEMPORAL_NODES_PATH = "../../data/dqr/new_temporal_nodes.csv"

nodes = pd.read_csv(TEMPORAL_NODES_PATH)
edges = pd.read_csv(TEMPORAL_EDGES_PATH)

def test_convergence(prev_values, curr_values, tol=1e-6):
    return abs(prev_values - curr_values).sum() < tol

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

def power_iteration(node_ids, incoming, out_degree, importance, damping=0.85, max_iter=100, tol=1e-6):
    N = len(node_ids)
    
    for iteration in range(max_iter):
        print("hi")
        prev_importance = importance.copy()
        new_importance = {}
        
        for i, node in enumerate(node_ids):
            if i % 1000 == 0:
                print(f"  Node {i}/{len(node_ids)}")
                
            rank_sum = sum(
                prev_importance[src] / out_degree[src] if out_degree.get(src, 0) > 0 else 0
                for src in incoming[node] if src in prev_importance
            )
            new_importance[node] = (1 - damping) / N + damping * rank_sum

        
        print("bye")

        for node in node_ids:
            importance.loc[node] = new_importance[node]

        diff = abs(prev_importance - importance).sum()
        if iteration % 10 == 0 or iteration < 10:
            print(f"Iteration {iteration + 1}: convergence diff = {diff:.8f} (target: {tol})")

        if test_convergence(prev_importance, importance, tol):
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return importance

def pagerank(nodes, edges, damping=0.85, max_iter=100, tol=1e-6):
    node_ids = nodes['node_id'].tolist()
    N = len(node_ids)
    importance = pd.Series(1.0 / N, index=node_ids)

    adjacency_df = edges.groupby('src')['dst'].apply(set)
    adjacency = {node: adjacency_df.get(node, set()) for node in node_ids}

    out_degree = {node: len(adjacency[node]) for node in node_ids}

    incoming_df = edges.groupby('dst')['src'].apply(list)
    incoming = {node: incoming_df.get(node, []) for node in node_ids}

    all_edge_nodes = set(edges['src'].unique()) | set(edges['dst'].unique())
    for edge_node in all_edge_nodes:
        if edge_node not in out_degree:
            out_degree[edge_node] = len(edges[edges['src'] == edge_node])

    

    importance = power_iteration(node_ids, incoming, out_degree, importance, damping, max_iter, tol)

    new_nodes = nodes.copy()
    new_nodes['importance'] = new_nodes['node_id'].map(importance)
    return new_nodes

new_nodes = pagerank(nodes, edges)

print(f"Score sum test passed: {test_score_sum(new_nodes)}")
show_score_distribution(new_nodes)

new_nodes.to_csv(NEW_TEMPORAL_NODES_PATH, index=False)
edges.to_csv(NEW_TEMPORAL_EDGES_PATH, index=False)
