import argparse
import os
from typing import List

from tgrag.construct_graph_scripts.process import process_graph
from tgrag.utils.path import get_scratch

parser = argparse.ArgumentParser(
    description='Construct and process graph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--slices',
    nargs='+',
    required=True,
    help='[Data Construction] List of CC time-slices to aggregate, e.g., --slices CC-MAIN-2017-13 CC-MAIN-2017-26',
)
parser.add_argument(
    '--min-deg',
    type=int,
    required=True,
    help='[Processing] Minimum degree threshold for nodes to keep.',
)
parser.add_argument(
    '--subnetworks',
    action='store_true',
    help='[Processing] Whether to create subnetworks centered from gold-standard label.',
)


# def handle_subnetworks(construct_subnetworks: bool, out_path: str) -> None:
#     if construct_subnetworks:
#         output_path = os.path.join(out_path, 'subnetworks')
#         pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#         construct_subnetwork(
#             dqr_path, output_path, temporal_edges_df, temporal_vertices_df
#         )


def main(
    slices: List[str], min_deg: int, only_targets: bool, subnetworks: bool
) -> None:
    for slice_id in slices:
        graph_path = os.path.join(get_scratch(), 'crawl-data', slice_id, 'output')
        process_graph(graph_path, min_deg, slice_id, only_targets)

        # if subnetworks:
        #     handle_subnetworks(subnetworks, graph_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        slices=args.slices,
        min_deg=args.min_deg,
        only_targets=False,
        subnetworks=args.subnetworks,
    )
