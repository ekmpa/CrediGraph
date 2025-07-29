import argparse
import glob
import gzip
import os

import tqdm

from tgrag.utils.data_loading import get_ids_from_set
from tgrag.utils.load_labels import get_labelled_set, get_target_set
from tgrag.utils.seed_set import get_seed_set


def append_edges(wanted_ids: set[str], source_base: str, target_base_root: str) -> None:
    """Append edges where both endpoints are in wanted_ids to target.
    Avoids duplicates.
    """
    source_dir = os.path.join(source_base, 'edges')
    target_path = os.path.join(target_base_root, 'edges.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')

    kept_edges = set()
    discarded = 0

    for source_file in matches:
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # skip malformed lines
                src, dst = parts[0], parts[1]
                if src in wanted_ids and dst in wanted_ids:
                    kept_edges.add(line.strip())
                else:
                    discarded += 1

    # Include existing target edges:
    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                kept_edges.add(line.strip())

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for edge in kept_edges:
            f.write(edge + '\n')

    print(f'[INFO] Kept {len(kept_edges)} edges → {target_path}')
    print(f'[INFO] Discarded {discarded} edges not connecting wanted IDs')


def append_nodes(
    endpoint_ids: set[str], source_base: str, target_base_root: str
) -> None:
    """Keep vertices with IDs in endpoint_ids.
    Append to existing aggregated vertices file.
    """
    source_dir = os.path.join(source_base, 'vertices')
    target_path = os.path.join(target_base_root, 'vertices.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')

    kept_vertices = []
    kept_ids = set()
    discarded = 0

    for source_file in matches:
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue  # skip malformed lines
                id_ = parts[0]
                if id_ in endpoint_ids:
                    kept_vertices.append(line)
                    kept_ids.add(id_)
                else:
                    discarded += 1

    # Include any existing target vertices:
    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            kept_vertices.extend(f.readlines())

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        f.writelines(kept_vertices)

    kept_vertices = list(set(kept_vertices))  # for duplicates

    print(f'[INFO] Kept {len(kept_vertices)} vertices → {target_path}')
    print(f'[INFO] Discarded {discarded} vertices that were not in seed set.')
    print(f'[INFO] Unique IDs in final vertices: {len(kept_ids)}')


def gradual_seed(source_base: str, target_base_root: str) -> None:
    """This append function works for the gradual building of the subgraph based on the seed set."""
    os.makedirs(target_base_root, exist_ok=True)

    wanted_domains = get_seed_set()

    wanted_ids = get_ids_from_set(wanted_domains, source_base)
    append_edges(wanted_ids, source_base, target_base_root)
    append_nodes(wanted_ids, source_base, target_base_root)


## BETWEEN HERE AND MAIN:
## These are old logics we can eventually delete.
## Keeping until we actually confirm our final construction


def gradual_full(source_base: str, target_base_root: str) -> None:
    """This append function works for gradual building of the entire graph."""
    os.makedirs(target_base_root, exist_ok=True)

    for subdir in ['edges', 'vertices']:
        source_dir = os.path.join(source_base, subdir)
        target_filename = f'{subdir}.txt.gz'
        target_path = os.path.join(target_base_root, target_filename)

        matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))
        if not matches:
            print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
            continue

        combined_data = []

        # Add all new data from source files
        for source_file in matches:
            with gzip.open(source_file, 'rt', encoding='utf-8') as f:
                combined_data.extend(f.readlines())

        # Add existing target data if target exists (i.e continue build of previous runs)
        if os.path.exists(target_path):
            with gzip.open(target_path, 'rt', encoding='utf-8') as f:
                combined_data.extend(f.readlines())

        with gzip.open(target_path, 'wt', encoding='utf-8') as f:
            f.writelines(combined_data)

        print(f'[INFO] Aggregated {len(matches)} file(s) → {target_path}')


def one_hop(wanted_ids: set[str], source_base: str, target_base_root: str) -> set[str]:
    """Keep edges with at least one endpoint in wanted_ids.
    Append to existing aggregated edges file.
    Return new set of all endpoint IDs in kept edges.
    """
    source_dir = os.path.join(source_base, 'edges')
    target_path = os.path.join(target_base_root, 'edges.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return set()

    kept_edges = []
    new_endpoint_ids = set()
    discarded = 0

    for source_file in matches:
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue  # skip malformed lines
                src, dst = parts
                if src in wanted_ids or dst in wanted_ids:
                    kept_edges.append(line)
                    new_endpoint_ids.update([src, dst])
                else:
                    discarded += 1

    # If existing target edges file exists, include its edges too:
    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            kept_edges.extend(f.readlines())

    kept_edges = list(set(kept_edges))  # for duplicates

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        f.writelines(kept_edges)

    print(f'[INFO] Kept {len(kept_edges)} edges → {target_path}')
    print(f'[INFO] Discarded {discarded} edges that were not connected to wanted set.')
    print(f'[INFO] Found {len(new_endpoint_ids)} endpoint IDs in kept edges')

    return new_endpoint_ids


def gradual_subset(source_base: str, target_base_root: str) -> None:
    """This append function works for the gradual building of the subgraph based on a 1-hop neighborhood of the wanted set."""
    os.makedirs(target_base_root, exist_ok=True)

    wanted_domains = get_target_set()
    wanted_domains.update(get_labelled_set())  # the target set may miss some labels

    wanted_ids = get_ids_from_set(wanted_domains, source_base)

    endpoint_ids = one_hop(wanted_ids, source_base, target_base_root)
    endpoint_ids.update(wanted_ids)  # to get non-connected nodes
    append_nodes(endpoint_ids, source_base, target_base_root)


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

    gradual_seed(args.source, args.target)


if __name__ == '__main__':
    main()
