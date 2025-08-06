import pandas as pd
import matplotlib.pyplot as plt

TEMPORAL_EDGES_PATH = "../../data/dqr/temporal_edges.csv"
TEMPORAL_NODES_PATH = "../../data/dqr/temporal_nodes.csv"

NEW_TEMPORAL_EDGES_PATH = "../../data/dqr/new_temporal_edges.csv"
NEW_TEMPORAL_NODES_PATH = "../../data/dqr/new_temporal_nodes.csv"

nodes = pd.read_csv(TEMPORAL_NODES_PATH)
edges = pd.read_csv(TEMPORAL_EDGES_PATH)

duplicate_nodes = nodes[nodes.duplicated(subset=['node_id'], keep=False)]
if not duplicate_nodes.empty:
    print(f"WARNING: Found {len(duplicate_nodes)} duplicate node IDs in {TEMPORAL_NODES_PATH}")
    print("Duplicate node IDs:")
    print(duplicate_nodes['node_id'].value_counts().sort_index())
    print("\nFirst few duplicate rows:")
    print(duplicate_nodes.head(10))
    
    original_count = len(nodes)
    nodes = nodes.drop_duplicates(subset=['node_id'], keep='first')
    print(f"\nRemoved {original_count - len(nodes)} duplicate rows")
    print(f"Unique nodes remaining: {len(nodes)}")
else:
    print(f"No duplicate node IDs found in {TEMPORAL_NODES_PATH}")

def test_convergence(prev_values, curr_values, tol=1e-6):
    return abs(prev_values - curr_values).sum() < tol

def test_score_sum(new_nodes, tol=1e-3):
    total = new_nodes['importance'].sum()
    print(f"Total importance score: {total}")
    print(f"Expected: 1.0, Difference: {abs(total - 1.0)}")
    return abs(total - 1.0) < tol

def show_score_distribution(new_nodes):
    plt.figure(figsize=(12, 8))
    
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
    
    print(f"\nPageRank Statistics:")
    print(f"Min score: {new_nodes['importance'].min():.8f}")
    print(f"Max score: {new_nodes['importance'].max():.8f}")
    print(f"Mean score: {new_nodes['importance'].mean():.8f}")
    print(f"Median score: {new_nodes['importance'].median():.8f}")
    print(f"95th percentile: {new_nodes['importance'].quantile(0.95):.8f}")
    print(f"99th percentile: {new_nodes['importance'].quantile(0.99):.8f}")

def power_iteration(node_ids, incoming, out_degree, importance, damping=0.85, max_iter=100, tol=1e-6):
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
    
    return importance

def pagerank(nodes, edges, damping=0.85, max_iter=100, tol=1e-6):
    nodes_from_df = set(nodes['node_id'])
    nodes_from_edges = set(edges['src'].unique()) | set(edges['dst'].unique())
    all_node_ids = nodes_from_df | nodes_from_edges
    
    missing_in_nodes = nodes_from_edges - nodes_from_df
    if missing_in_nodes:
        print(f"WARNING: Found {len(missing_in_nodes)} nodes in edges that are not in nodes DataFrame")
        print(f"First few missing nodes: {list(missing_in_nodes)[:10]}")
    
    node_ids = list(all_node_ids)
    N = len(node_ids)
    importance = pd.Series(1.0 / N, index=node_ids)

    adjacency_df = edges.groupby('src')['dst'].apply(set)
    adjacency = {node: adjacency_df.get(node, set()) for node in node_ids}

    out_degree = {node: len(adjacency[node]) for node in node_ids}

    incoming_df = edges.groupby('dst')['src'].apply(list)
    incoming = {node: incoming_df.get(node, []) for node in node_ids}

    importance = power_iteration(node_ids, incoming, out_degree, importance, damping, max_iter, tol)

    result_nodes = pd.DataFrame({'node_id': node_ids})
    result_nodes['importance'] = result_nodes['node_id'].map(importance)
    
    if len(nodes.columns) > 1:  # If nodes has more than just node_id
        result_nodes = result_nodes.merge(nodes, on='node_id', how='left')
    
    return result_nodes

new_nodes = pagerank(nodes, edges)
print(f"Score sum test passed: {test_score_sum(new_nodes)}")
show_score_distribution(new_nodes)
new_nodes.to_csv(NEW_TEMPORAL_NODES_PATH, index=False)
edges.to_csv(NEW_TEMPORAL_EDGES_PATH, index=False)
