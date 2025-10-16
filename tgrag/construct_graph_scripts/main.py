import argparse

from tgrag.construct_graph_scripts.process import process_graph

parser = argparse.ArgumentParser(
    description='Process graph and get graph stats.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--graph-path',
    required=True,
    help='Folder containing edges.txt.gz (+ optional vertices.txt[.gz])',
)
parser.add_argument('--slice', required=True, help='Crawl slice like CC-MAIN-YYYY-WW')
parser.add_argument(
    '--min-deg',
    type=int,
    required=True,
    help='Minimum degree threshold for nodes to keep (>=).',
)
parser.add_argument('--mem', default='60%', help='Sort memory (default: 60%)')


def main(
    graph_path: str,
    slice: str,
    min_deg: int,
    mem: str,
) -> None:
    # TODO: go back to multi-slice.
    # Right now, need to feed graph path explicitly for current set-up with multiple bash_scripts folders,
    # This will change when we standardize that folder structure to optimize the pipeline.

    # for slice_id in slices:
    #     graph_path = os.path.join(get_scratch(), 'crawl-data', slice_id, 'output')
    #     process_graph(graph_path, min_deg, slice_id, only_targets)

    process_graph(graph_path, slice, min_deg, mem)


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        graph_path=args.graph_path,
        slice=args.slice,
        min_deg=args.min_deg,
        mem=args.mem,
    )
