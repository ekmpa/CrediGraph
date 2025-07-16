import pathlib
import re

import pandas as pd
from tqdm import tqdm


def contains_base_domain(input: str, base_domain: str) -> bool:
    forward_pat = rf'\b(?:[\w-]+\.)*{re.escape(base_domain)}\b'

    reversed_base = '.'.join(base_domain.split('.')[::-1])  # nasa.gov -> gov.nasa
    reverse_pat = rf'\b{re.escape(reversed_base)}(?:\.[\w-]+)*\b'

    combined_pat = rf'(?:{forward_pat}|{reverse_pat})'

    return re.search(combined_pat, input) is not None


def construct_subnetwork(
    dqr_path: str,
    output_path: str,
    temporal_edges: pd.DataFrame,
    temporal_vertices: pd.DataFrame,
    n_hop: int = 1,
) -> None:
    df_dqr = pd.read_csv(dqr_path)

    for base_domain, _ in tqdm(
        zip(df_dqr['domain'], df_dqr['pc1']), desc='Labelled Domain'
    ):
        matched_vertices = temporal_vertices[
            temporal_vertices['domain'].apply(
                lambda d: contains_base_domain(d, base_domain)
            )
        ]

        if matched_vertices.empty:
            continue

        seen_node_ids = set(matched_vertices['node_id'])
        current_frontier = set(seen_node_ids)
        all_edges = pd.DataFrame(columns=temporal_edges.columns)

        for _ in range(n_hop + 1):
            if not current_frontier:
                break

            hop_edges = temporal_edges[
                temporal_edges['src'].isin(current_frontier)
                | temporal_edges['dst'].isin(current_frontier)
            ]

            all_edges = pd.concat([all_edges, hop_edges], ignore_index=True)

            hop_node_ids = set(hop_edges['src']).union(set(hop_edges['dst']))

            next_frontier = hop_node_ids - seen_node_ids

            seen_node_ids.update(next_frontier)
            current_frontier = next_frontier

        sub_vertices_df = temporal_vertices[
            temporal_vertices['node_id'].isin(seen_node_ids)
        ]
        sub_edges_df = all_edges.drop_duplicates(subset=['src', 'dst'])

        safe_domain = base_domain.replace('.', '_')
        sub_path = pathlib.Path(output_path) / f'sub_vertices_domain_{safe_domain}'
        sub_path.mkdir(parents=True, exist_ok=True)

        sub_vertices_df.to_csv(sub_path / 'vertices.csv', index=False)
        sub_edges_df.to_csv(sub_path / 'edges.csv', index=False)
