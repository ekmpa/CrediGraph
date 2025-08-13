# Generate a subgraph of a given graph.
# At the same time, it maps node IDs to domain names.
# Options:
#   - Deg > k
#   - With or without non-connected nodes

import argparse
import gzip
import os
from typing import Dict, Set, Tuple


def count_lines(path: str) -> int:  # move to utils
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return sum(1 for _ in f)


def add_node_IDs(graph_path: str) -> Tuple[Set[str], Set[str], Dict[int, int]]:
    domain_to_id: Dict[str, int] = {}
    vert_path = os.path.join(graph_path, 'vertices.txt.gz')
    edges_path = os.path.join(graph_path, 'edges.txt.gz')

    print(
        f'[INFO] Initial graph has {count_lines(vert_path)} vertices and {count_lines(edges_path)} edges.'
    )

    ID_count = 0
    new_verts: Set[str] = set()
    ID_to_deg: Dict[int, int] = {}

    with gzip.open(vert_path, 'rt', encoding='utf-8') as f:
        for line in f:
            domain = line.strip()
            if domain in domain_to_id:
                ID = domain_to_id[domain]

            else:
                ID = ID_count
                domain_to_id[domain] = ID
                ID_count += 1

            line = f'{ID}\t{domain}'
            new_verts.add(line)
            ID_to_deg[ID] = 0

    # max ID = ID count = # unique domains should = # vertices
    print(f'max ID: {ID_count}')

    new_edges: Set[str] = set()

    with gzip.open(edges_path, 'rt', encoding='utf-8') as f:
        for line in f:
            src_t, dst_t = line.strip().split('\t')
            if src_t not in domain_to_id.keys() or dst_t not in domain_to_id.keys():
                continue
            src, dst = int(domain_to_id[src_t]), int(domain_to_id[dst_t])

            ID_to_deg[src] += 1
            ID_to_deg[dst] += 1

            line = f'{src}\t{dst}'
            new_edges.add(line)

    return new_verts, new_edges, ID_to_deg


def discard_deg(
    output_path: str,
    min_deg: int,
    new_verts: Set[str],
    new_edges: Set[str],
    ID_to_deg: Dict[int, int],
) -> None:
    vert_output = os.path.join(output_path, 'vertices.txt.gz')
    IDs_to_keep: Set[int] = set()

    with gzip.open(vert_output, 'wt', encoding='utf-8') as f:
        for vertex in new_verts:
            ID = int(vertex.split('\t')[0])
            if ID_to_deg[ID] > min_deg:
                IDs_to_keep.add(ID)
                f.write(vertex + '\n')

    edges_output = os.path.join(output_path, 'edges.txt.gz')
    with gzip.open(edges_output, 'wt', encoding='utf-8') as f:
        for edge in new_edges:
            src, dst = edge.split('\t')
            if int(src) in IDs_to_keep and int(dst) in IDs_to_keep:
                f.write(f'{src}\t{dst}\n')

    print(
        f'Having discarded all nodes with deg <= {min_deg} we have {count_lines(vert_output)} vertices and {count_lines(edges_output)} edges.'
    )


def generate(graph_path: str, min_deg: int) -> None:
    output_path = os.path.join(graph_path, f'deg>{min_deg}')
    os.makedirs(output_path, exist_ok=True)

    new_verts, new_edges, ID_to_deg = add_node_IDs(graph_path)

    discard_deg(output_path, min_deg, new_verts, new_edges, ID_to_deg)

    # discard deg < min deg

    # and generate targets.csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Subgraph generation.')
    parser.add_argument(
        '--min-deg',
        type=int,
        required=True,
        help='Minimum degree threshold for nodes to keep.',
    )
    parser.add_argument(
        '--graph-path',
        type=str,
        required=True,
        help='Path to CC output/ folder with the vertex and edge files.',
    )
    args = parser.parse_args()

    print(f'[START] Running subgraph generation with deg > {args.min_deg}')
    generate(args.graph_path, args.min_deg)


if __name__ == '__main__':
    main()
