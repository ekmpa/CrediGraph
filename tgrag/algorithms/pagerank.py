from pagerank_utils import *

TEMPORAL_EDGES_PATH = "../../data/dqr/temporal_edges.csv"
TEMPORAL_NODES_PATH = "../../data/dqr/temporal_nodes.csv"

NEW_TEMPORAL_EDGES_PATH = "../../data/dqr/new_temporal_edges.csv"
NEW_TEMPORAL_NODES_PATH = "../../data/dqr/new_temporal_nodes.csv"

damping, max_iter, tol = 0.85, 100, 1e-6
nodes, edges = preprocess_data(TEMPORAL_NODES_PATH, TEMPORAL_EDGES_PATH)
node_ids, adjacency, out_degree, incoming = build_adjacency(nodes, edges)

def compute_new_importance(node_ids, prev_importance, out_degree, incoming, damping, dangling_sum, N):
    new_importance = {}
    for node in node_ids:
        rank_sum = 0
        for src in incoming[node]:
            if src in prev_importance and out_degree.get(src, 0) > 0:
                rank_sum += prev_importance[src] / out_degree[src]
        dangling_contribution = dangling_sum / N
        
        #   (1 - damping): probability of jumping to a random page
        new_importance[node] = (1 - damping) / N + damping * (rank_sum + dangling_contribution)
    return pd.Series(new_importance)

#   Run PageRank algorithm
N = len(node_ids)
importance = pd.Series(1.0 / N, index=node_ids)
iteration = 0
converged = False
while iteration < max_iter and not converged:
    prev_importance = importance.copy()
    dangling_sum = sum(prev_importance[node] for node in node_ids if out_degree.get(node, 0) == 0)
    importance = compute_new_importance(node_ids, prev_importance, out_degree, incoming, damping, dangling_sum, N)
    converged = check_convergence(iteration, prev_importance, importance, tol)
    iteration += 1

#   Create result DataFrame
nodes['importance'] = nodes['node_id'].map(importance)

#   Run tests
convergence_test = converged
score_sum = test_score_sum(nodes)
positive_values_test = test_positive_values(nodes)
degree_correlation_test = test_degree_correlation(nodes, edges)

print(f"Convergence test passed: {convergence_test}")
print(f"Score sum test passed: {score_sum}")
print(f"Positive values test passed: {positive_values_test}")
print(f"Degree correlation test passed: {degree_correlation_test}")

show_score_distribution(nodes)

nodes.to_csv(NEW_TEMPORAL_NODES_PATH, index=False)
edges.to_csv(NEW_TEMPORAL_EDGES_PATH, index=False)
print(f"Results saved to {NEW_TEMPORAL_NODES_PATH} and {NEW_TEMPORAL_EDGES_PATH}")