import argparse
import glob
import gzip
import os

from tgrag.utils.data_loading import get_ids_from_set
from tgrag.utils.load_labels import get_labelled_set


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

        # TO DO: Test: deduplicate lines
        # combined_data = list(set(combined_data))

        with gzip.open(target_path, 'wt', encoding='utf-8') as f:
            f.writelines(combined_data)

        print(f'[INFO] Aggregated {len(matches)} file(s) → {target_path}')


def append_edges(
    wanted_ids: set[str], source_base: str, target_base_root: str
) -> set[str]:
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

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        f.writelines(kept_edges)

    print(f'[INFO] Kept {len(kept_edges)} edges → {target_path}')
    print(f'[INFO] Discarded {discarded} edges that were not connected to wanted set.')
    print(f'[INFO] Found {len(new_endpoint_ids)} endpoint IDs in kept edges')

    return new_endpoint_ids


def append_nodes(
    endpoint_ids: set[str], source_base: str, target_base_root: str
) -> int:
    """Keep vertices with IDs in endpoint_ids.
    Append to existing aggregated vertices file.
    Return the count of unique nodes kept.
    """
    source_dir = os.path.join(source_base, 'vertices')
    target_path = os.path.join(target_base_root, 'vertices.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return 0

    kept_vertices = []
    kept_ids = set()
    discarded = 0

    for source_file in matches:
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for line in f:
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

    print(f'[INFO] Kept {len(kept_vertices)} vertices → {target_path}')
    print(
        f'[INFO] Discarded {discarded} vertices that were not in 1-hop neighborhood of wanted set.'
    )
    print(f'[INFO] Unique IDs in final vertices: {len(kept_ids)}')

    return len(kept_ids)


def gradual_sampled(source_base: str, target_base_root: str) -> None:
    """This append function works for the gradual building of the subgraph based on a 1-hop neighborhood of the wanted set.
    For now, the 'wanted set' is just our labelled nodes. To be supplemented later.
    """
    os.makedirs(target_base_root, exist_ok=True)

    wanted_domains = get_labelled_set()
    wanted_ids = get_ids_from_set(wanted_domains, source_base)

    # TO DO: aggregate wanted_ids with set from PR/HC sampling (Seb)
    # Currently, just labelled set

    # TO DO: deduplicate lines

    endpoint_ids = append_edges(wanted_ids, source_base, target_base_root)
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

    gradual_sampled(args.source, args.target)


if __name__ == '__main__':
    main()
