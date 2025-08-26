import argparse
import os
import pathlib
from typing import List

import pandas as pd

from tgrag.construct_graph_scripts.subnetwork_construct import (
    construct_subnetwork,
)
from tgrag.utils.load_labels import get_credibility_intersection
from tgrag.utils.path import get_crawl_data_path, get_data_paths, get_root_dir

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


def main(slices: List[str] | str, construct_subnetworks: bool) -> None:
    base_path = get_root_dir()
    crawl_path = get_crawl_data_path(base_path)

    for slice_id in slices:
        vertices_path, edges_path = get_data_paths(slice_id, crawl_path)

        get_credibility_intersection(
            data_path=f'{crawl_path}/{slice_id}',
            label_path=base_path,
            time_slice=slice_id,
        )

        annotated_vertices_path = os.path.join(
            f'{crawl_path}/{slice_id}/output_text_dir/', 'vertices.csv'
        )

    dqr_path = f'{base_path}/data/dqr/domain_pc1.csv'
    # temporal_path = f'{base_path}/data/crawl-data/temporal'
    temporal_path = os.path.join(
        os.environ.get('SCRATCH', f'{base_path}/data'), 'crawl-data', 'temporal'
    )
    temporal_edges_df = pd.read_csv(f'{temporal_path}/temporal_edges.csv')
    temporal_vertices_df = pd.read_csv(f'{temporal_path}/temporal_nodes.csv')

    if construct_subnetworks:
        output_path = f'{base_path}/data/crawl-data/sub-networks/'
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        construct_subnetwork(
            dqr_path, output_path, temporal_edges_df, temporal_vertices_df
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.slices, args.subnetworks)
