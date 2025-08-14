import pandas as pd
import matplotlib.pyplot as plt

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

def preprocess_data(nodes_path, edges_path):
    print("Starting data preprocessing...")
    
    # Load raw data
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # Handle duplicate nodes
    duplicate_nodes = nodes[nodes.duplicated(subset=['node_id'], keep=False)]
    if not duplicate_nodes.empty:
        print(f"WARNING: Found {len(duplicate_nodes)} duplicate node IDs in {nodes_path}")
        print("Duplicate node IDs:")
        print(duplicate_nodes['node_id'].value_counts().sort_index())
        print("\nFirst few duplicate rows:")
        print(duplicate_nodes.head(10))
        
        original_count = len(nodes)
        nodes = nodes.drop_duplicates(subset=['node_id'], keep='first')
        print(f"\nRemoved {original_count - len(nodes)} duplicate rows")
        print(f"Unique nodes remaining: {len(nodes)}")
    else:
        print(f"No duplicate node IDs found in {nodes_path}")
    
    # Validate edge consistency
    nodes_from_df = set(nodes['node_id'])
    nodes_from_edges = set(edges['src'].unique()) | set(edges['dst'].unique())
    
    missing_in_nodes = nodes_from_edges - nodes_from_df
    if missing_in_nodes:
        print(f"WARNING: Found {len(missing_in_nodes)} nodes in edges that are not in nodes DataFrame")
        print(f"First few missing nodes: {list(missing_in_nodes)[:10]}")
        
        # Add missing nodes to nodes DataFrame
        missing_nodes_df = pd.DataFrame({'node_id': list(missing_in_nodes)})
        nodes = pd.concat([nodes, missing_nodes_df], ignore_index=True)
        print(f"Added {len(missing_in_nodes)} missing nodes to nodes DataFrame")
    
    # Remove self-loops (optional)
    self_loops = edges[edges['src'] == edges['dst']]
    if not self_loops.empty:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        edges = edges[edges['src'] != edges['dst']]
    
    # Remove duplicate edges (optional)
    original_edge_count = len(edges)
    edges = edges.drop_duplicates(subset=['src', 'dst'], keep='first')
    if len(edges) < original_edge_count:
        print(f"Removed {original_edge_count - len(edges)} duplicate edges")
    
    print(f"Preprocessing complete: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges

def build_adjacency(nodes, edges):
    all_node_ids = set(nodes['node_id'])
    node_ids = list(all_node_ids)
    N = len(node_ids)

    adjacency_df = edges.groupby('src')['dst'].apply(set)
    adjacency = {node: adjacency_df.get(node, set()) for node in node_ids}

    out_degree = {node: len(adjacency[node]) for node in node_ids}

    incoming_df = edges.groupby('dst')['src'].apply(list)
    incoming = {node: incoming_df.get(node, []) for node in node_ids}

    return node_ids, N, adjacency, out_degree, incoming