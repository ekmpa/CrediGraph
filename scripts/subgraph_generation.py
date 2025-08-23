# Generate a subgraph of a given graph.
# At the same time, it maps node IDs to domain names.
# Options:
#   - Deg > k

# TODO : either code generate_fused if we want it or (most likely) remove that option altogether

import argparse
import csv
import gzip
import os
from typing import Dict, List, Optional, Set, Tuple

from tgrag.utils.data_loading import count_lines, iso_week_to_timestamp
from tgrag.utils.load_labels import get_full_dict


def add_node_IDs(graph_path: str) -> Tuple[Set[str], Set[str], Dict[int, int]]:
    domain_to_id: Dict[str, int] = {}
    vert_path = os.path.join(graph_path, 'vertices.txt.gz')
    edges_path = os.path.join(graph_path, 'edges.txt.gz')

    V = count_lines(vert_path)
    E = count_lines(edges_path)

    # dir_sparsity = (E / (V * (V - 1))) if V > 1 else 0.0
    undir_sparsity = (E / (V * (V - 1) / 2)) if V > 1 else 0.0
    print(
        f'[INFO] Initial graph: V={V}, E={E} \n \t undirected sparsity={undir_sparsity:.6g}'
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
    # STATS
    V0 = len(new_verts)
    E0 = len(new_edges)
    deg_values = ID_to_deg.values()
    iso_count = sum(1 for d in deg_values if d == 0)
    leaf_count = sum(1 for d in ID_to_deg.values() if d == 1)
    min_deg_all = min(deg_values) if V0 else 0
    max_deg_all = max(deg_values) if V0 else 0
    mean_deg_all = (2 * E0 / V0) if V0 else 0.0  # matches total-degree convention used
    self_loops_all = 0
    for e in new_edges:
        s, t = e.split('\t')
        if s == t:
            self_loops_all += 1

    vert_output = os.path.join(output_path, 'vertices.txt.gz')
    IDs_to_keep: Set[int] = set()

    with gzip.open(vert_output, 'wt', encoding='utf-8') as f:
        for vertex in new_verts:
            ID = int(vertex.split('\t')[0])
            if ID_to_deg[ID] > min_deg:
                IDs_to_keep.add(ID)
                f.write(vertex + '\n')

    edges_output = os.path.join(output_path, 'edges.txt.gz')
    kept_self_loops = 0
    kept_edges = 0
    kept_deg: Dict[int, int] = {i: 0 for i in IDs_to_keep}  # degrees in filtered graph
    with gzip.open(edges_output, 'wt', encoding='utf-8') as f:
        for edge in new_edges:
            src_t, dst_t = edge.split('\t')
            src = int(src_t)
            dst = int(dst_t)
            if src in IDs_to_keep and dst in IDs_to_keep:
                kept_edges += 1
                if src == dst:
                    kept_self_loops += 1
                    kept_deg[src] += 2
                else:
                    kept_deg[src] += 1
                    kept_deg[dst] += 1
                f.write(f'{src_t}\t{dst_t}\n')

    # STATS (POST)
    V = count_lines(vert_output)
    E = count_lines(edges_output)
    (E / (V * (V - 1))) if V > 1 else 0.0
    undir_sparsity = (E / (V * (V - 1) / 2)) if V > 1 else 0.0

    removed_V = V0 - V
    removed_E = E0 - E
    kept_V_pct = (100.0 * V / V0) if V0 else 0.0
    kept_E_pct = (100.0 * E / E0) if E0 else 0.0

    kept_deg_vals = [ID_to_deg[i] for i in IDs_to_keep]
    min_deg_kept = min(kept_deg_vals)
    max_deg_kept = max(kept_deg_vals)
    mean_deg_kept = (2 * E / V) if V else 0.0
    iso_kept = sum(1 for d in kept_deg_vals if d == 0)
    leaf_kept = sum(1 for d in kept_deg_vals if d == 1)
    # print(f'[INFO] Having discarded all nodes with deg <= {min_deg}, graph: V={V}, E={E}, directed sparsity={dir_sparsity:.6g} & undirected sparsity={undir_sparsity:.6g}')

    print('[INFO] Pre-filter:')
    print(f'       V0={V0}, E0={E0}, iso={iso_count}, leaves={leaf_count}')
    print(f'       degree(min/mean/max)={min_deg_all}/{mean_deg_all:.4g}/{max_deg_all}')
    print(f'       self-loops={self_loops_all}')

    print(f'[INFO] Having discarded nodes with deg <= {min_deg}:')
    print(f'       V={V} ({kept_V_pct:.2f}% kept), E={E} ({kept_E_pct:.2f}% kept)')
    print(f'       removed: vertices={removed_V}, edges={removed_E}')
    print(f'       undirected sparsity={undir_sparsity:.6g}')
    print(f'       kept self-loops={kept_self_loops}')
    print(
        f'       kept degree(min/mean/max)={min_deg_kept}/{mean_deg_kept:.4g}/{max_deg_kept}'
    )
    print(f'       kept iso={iso_kept}, kept leaves={leaf_kept}')


def add_timestamps(output_path: str, slice: str) -> None:
    ts = iso_week_to_timestamp(slice)

    vertices_in = os.path.join(output_path, 'vertices.txt.gz')
    edges_in = os.path.join(output_path, 'edges.txt.gz')
    vertices_out = os.path.join(output_path, 'vertices.csv.gz')
    edges_out = os.path.join(output_path, 'edges.csv.gz')

    # vertices: nid\tdomain -> nid,domain,ts
    with (
        gzip.open(vertices_in, 'rt', encoding='utf-8') as fin,
        gzip.open(vertices_out, 'wt', encoding='utf-8', newline='') as fout,
    ):
        fout.write('nid,domain,ts\n')
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            nid, domain = parts[0].strip(), parts[1].strip()
            fout.write(f'{nid},{domain},{ts}\n')

    # edges: src\tdst -> src,dst,ts
    with (
        gzip.open(edges_in, 'rt', encoding='utf-8') as fin,
        gzip.open(edges_out, 'wt', encoding='utf-8', newline='') as fout,
    ):
        fout.write('src,dst,ts\n')
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            src, dst = parts[0].strip(), parts[1].strip()
            fout.write(f'{src},{dst},{ts}\n')

    print(f'[INFO] Wrote CSVs with timestamps to {vertices_out} and {edges_out}')


def lookup(domain: str, dqr_domains: Dict[str, List[float]]) -> Optional[List[float]]:
    domain_parts = domain.split('.')
    for key, value in dqr_domains.items():
        key_parts = key.split('.')
        if key_parts[0] in domain_parts and key_parts[1] in domain_parts:
            return value
    return None


def generate_separate(
    output_path: str, vertices: str, dqr_domains: Dict[str, List[float]]
) -> None:
    targets: Dict[int, List[float]] = {}

    with gzip.open(vertices, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split(',')
            # print(parts[1].strip())
            result = lookup(parts[1].strip(), dqr_domains)
            if result is not None:  # parts[1].strip() in dqr_domains.keys():
                targets[int(parts[0].strip())] = result  # dqr_domains[parts[1].strip()]

    targets_path = os.path.join(output_path, 'targets.csv')

    with open(targets_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                'nid',
                'pc1',
                'afm',
                'afm_bias',
                'afm_min',
                'afm_rely',
                'fc',
                'mbfc',
                'mbfc_bias',
                'mbfc_fact',
                'mbfc_min',
                'lewandowsky_acc',
                'lewandowsky_trans',
                'lewandowsky_rely',
                'lewandowsky_mean',
                'lewandowsky_min',
                'misinfome_bin',
            ]
        )
        for key, value in targets.items():
            writer.writerow([key, value])

    print(f'Wrote targets to {targets_path}')


def generate_fused(
    output_path: str, vertices: str, dqr_dict: Dict[str, List[float]]
) -> None:
    print('TODO')


def generate_targets(output_path: str, separate_targets: bool) -> None:
    vertices = os.path.join(output_path, 'vertices.csv.gz')
    os.path.join(output_path, 'edges.csv.gz')
    # ßdqr_dict = get_labelled_dict()
    dqr_dict = get_full_dict()

    if separate_targets:
        generate_separate(output_path, vertices, dqr_dict)
    else:
        generate_fused(output_path, vertices, dqr_dict)


def generate(graph_path: str, min_deg: int, separate_targets: bool, slice: str) -> None:
    suffix = '_separate_targets' if separate_targets else ''
    rest = f'deg{min_deg}{suffix}'
    output_path = os.path.join(graph_path, rest)  # f"deg>{min_deg}_{suffix}")
    os.makedirs(output_path, exist_ok=True)

    new_verts, new_edges, ID_to_deg = add_node_IDs(graph_path)

    discard_deg(output_path, min_deg, new_verts, new_edges, ID_to_deg)

    add_timestamps(output_path, slice)  # makes them csv

    generate_targets(output_path, separate_targets)


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
    parser.add_argument('--slice', type=str, required=True, help='CC-MAIN-XXXX-XX')
    parser.add_argument(
        '--separate-targets',
        action='store_true',
        help='Set --separate-targets to save the ground truth labels in a separate targets.csv instead of as node features.',
    )
    args = parser.parse_args()

    print(f'[START] Running subgraph generation with deg > {args.min_deg}')
    generate(args.graph_path, args.min_deg, args.separate_targets, args.slice)


if __name__ == '__main__':
    main()
