import argparse
import os

from tgrag.utils.data_io import append_target_edges, append_target_nodes


def gradual_all(source_base: str, target_base_root: str) -> None:
    os.makedirs(target_base_root, exist_ok=True)
    id_to_domain = append_target_nodes(source_base, target_base_root)
    append_target_edges(source_base, target_base_root, id_to_domain)


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
