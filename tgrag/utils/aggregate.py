import argparse

from tgrag.construct_graph_scripts.process_compressed_text import (
    move_and_append_compressed_outputs,
)


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

    move_and_append_compressed_outputs(args.source, args.target)


if __name__ == '__main__':
    main()
