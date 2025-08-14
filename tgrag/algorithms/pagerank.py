from pagerank_utils import *

TEMPORAL_EDGES_PATH = "../../data/dqr/temporal_edges.csv"
TEMPORAL_NODES_PATH = "../../data/dqr/temporal_nodes.csv"

NEW_TEMPORAL_EDGES_PATH = "../../data/dqr/new_temporal_edges.csv"
NEW_TEMPORAL_NODES_PATH = "../../data/dqr/new_temporal_nodes.csv"

damping, max_iter, tol = 0.85, 100, 1e-6

nodes, edges = preprocess_data(TEMPORAL_NODES_PATH, TEMPORAL_EDGES_PATH)

# def power_iteration(node_ids, incoming, out_degree, importance, damping=damping, max_iter=max_iter, tol=tol):
    

node_ids, N, adjacency, out_degree, incoming = build_adjacency(nodes, edges)


#   Run PageRank algorithm
importance = pd.Series(1.0 / N, index=node_ids)
N = len(node_ids)
    
for iteration in range(max_iter):
    prev_importance = importance.copy()
    new_importance = {}
    
    dangling_sum = sum(
        prev_importance[node] for node in node_ids 
        if out_degree.get(node, 0) == 0
    )
    
    for i, node in enumerate(node_ids):
        rank_sum = sum(
            prev_importance[src] / out_degree[src] 
            for src in incoming[node] 
            if src in prev_importance and out_degree.get(src, 0) > 0
        )
        
        dangling_contribution = dangling_sum / N
        
        new_importance[node] = (1 - damping) / N + damping * (rank_sum + dangling_contribution)

    for node in node_ids:
        importance.loc[node] = new_importance[node]

    diff = abs(prev_importance - importance).sum()
    if iteration % 10 == 0 or iteration < 10:
        print(f"Iteration {iteration + 1}: convergence diff = {diff:.8f} (target: {tol})")

    if test_convergence(prev_importance, importance, tol):
        print(f"Converged after {iteration + 1} iterations")
        break
    
result_nodes = pd.DataFrame({'node_id': node_ids})
result_nodes['importance'] = result_nodes['node_id'].map(importance)

if len(nodes.columns) > 1:  # If nodes has more than just node_id
    result_nodes = result_nodes.merge(nodes, on='node_id', how='left')

print(f"Score sum test passed: {test_score_sum(result_nodes)}")
show_score_distribution(result_nodes)
result_nodes.to_csv(NEW_TEMPORAL_NODES_PATH, index=False)
edges.to_csv(NEW_TEMPORAL_EDGES_PATH, index=False)
print(f"Results saved to {NEW_TEMPORAL_NODES_PATH} and {NEW_TEMPORAL_EDGES_PATH}")