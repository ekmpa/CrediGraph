import argparse
import glob
import gzip
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set

from tgrag.utils.data_loading import read_edge_file, read_vertex_file


def append_all_nodes(source_base: str, target_base_root: str) -> Dict[int, int]:
    source_dir: str = os.path.join(source_base, 'vertices')
    target_path: str = os.path.join(target_base_root, 'vertices.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return {}

    combined_vertices: Set[str] = set()
    all_IDs: Set[int] = set()

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                combined_vertices.add(line)
                all_IDs.add(int(line.split('\t')[0]))

    max_ID = max(all_IDs) if all_IDs else 0
    old_to_new: Dict[int, int] = {}

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_vertex_file, matches))
        for result in results:
            for line in result:
                node_id = int(line.split('\t')[0])
                if node_id in all_IDs:
                    new_ID = max_ID
                    max_ID += 1
                    old_to_new[node_id] = new_ID
                    domain = line.split('\t')[1]
                    line = f'{new_ID}\t{domain}'
                combined_vertices.add(line)

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for vertex in combined_vertices:
            f.write(vertex + '\n')

    print(f'[INFO] Aggregated {len(combined_vertices)} vertices → {target_path}')
    return old_to_new


def append_all_edges(
    source_base: str, target_base_root: str, old_to_new: Dict[int, int]
) -> None:
    source_dir: str = os.path.join(source_base, 'edges')
    target_path: str = os.path.join(target_base_root, 'edges.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return

    combined_edges: Set[str] = set()

    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda path: read_edge_file(path, old_to_new), matches)
        )
        for result in results:
            combined_edges.update(result)

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                combined_edges.add(line.strip())

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for edge in combined_edges:
            f.write(edge + '\n')

    print(f'[INFO] Aggregated {len(combined_edges)} edges → {target_path}')


def gradual_all(source_base: str, target_base_root: str) -> None:
    """Gradual construction of full graph, no seed set, unique IDs."""
    os.makedirs(target_base_root, exist_ok=True)
    old_to_new: Dict[int, int] = append_all_nodes(source_base, target_base_root)
    append_all_edges(source_base, target_base_root, old_to_new)


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
