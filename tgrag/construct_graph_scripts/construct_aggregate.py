import argparse
import glob
import gzip
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Set, Tuple

from tgrag.utils.data_loading import read_edge_file, read_vertex_file


def append_all_nodes(source_base: str, target_base_root: str) -> Dict[int, str]:
    source_dir: str = os.path.join(source_base, 'vertices')
    target_path: str = os.path.join(target_base_root, 'vertices.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return {}

    combined_vertices: Set[str] = set()
    id_to_domain: Dict[int, str] = {}

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                combined_vertices.add(line)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(read_vertex_file, p) for p in matches]
        for fut in as_completed(futures):
            result = fut.result()
            for line in result:
                parts = line.split('\t')
                domain = parts[1]
                combined_vertices.add(domain)
                id_to_domain[int(parts[0])] = domain

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for vertex in combined_vertices:
            f.write(vertex + '\n')

    print(f'[INFO] Aggregated {len(combined_vertices)} vertices → {target_path}')
    return id_to_domain


def append_all_edges(
    source_base: str, target_base_root: str, id_to_domain: Dict[int, str]
) -> None:
    source_dir: str = os.path.join(source_base, 'edges')
    target_path: str = os.path.join(target_base_root, 'edges.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return

    combined_edges: Set[Tuple[str, str]] = set()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(read_edge_file, p, id_to_domain) for p in matches]
        for fut in as_completed(futures):
            combined_edges.update(fut.result())

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t', 1)
                combined_edges.add((parts[0], parts[1]))

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for src, dst in combined_edges:
            f.write(f'{src}\t{dst}\n')

    print(f'[INFO] Aggregated {len(combined_edges)} edges → {target_path}')


def gradual_all(source_base: str, target_base_root: str) -> None:
    os.makedirs(target_base_root, exist_ok=True)
    id_to_domain = append_all_nodes(source_base, target_base_root)
    append_all_edges(source_base, target_base_root, id_to_domain)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate and append compressed outputs (edges and vertices).'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Source base directory containing the new output subdirectories (edges/vertices).',
    )
    parser.add_argument(
        '--target',
        type=str,
        help='Target base directory where the aggregated output will be stored.',
    )

    args = parser.parse_args()

    gradual_all(args.source, args.target)


if __name__ == '__main__':
    main()
