import argparse
import logging

from tgrag.construct_graph_scripts.merge_dqr_ratings_trie_filter import (
    merge_dqr_to_node_parallel,
)
from tgrag.construct_graph_scripts.txt_to_csv_sqlite import (
    merge_vertices_rank_centrality,
)
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Baseline Dataset Construction',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--baseline-folder',
    type=str,
    default='data/baseline/',
    help='Folder containing *.vertices.txt, *.edges.txt and *.ranks.txt files',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='script_correlation.log',
    help='Name of log file at project root.',
)


def construct_baseline() -> None:
    args = parser.parse_args()
    root = get_root_dir()
    dqr_path = f'{root}/data/dqr/domain_pc1.csv'
    folder_path = f'{args.baseline_folder}'
    setup_logging(args.log_file)
    logging.info('***Merging Rank/Centrality with vertices.txt***')
    logging.info('***Converting txt to csv ***')
    merge_vertices_rank_centrality(folder_path)
    logging.info('***Merge DQR gold-standard labels***')
    node_path = f'{args.baseline_folder}/cc_baseline_nodes.csv'
    output_path = f'{args.baseline_folder}/cc_baseline_nodes_scored.csv'
    edges_path = f'{args.baseline_folder}/cc_baseline_edges.csv'
    filter_edges_output_path = f'{args.baseline_folder}/cc-baseline_edges_filtered.csv'
    workers = 16
    merge_dqr_to_node_parallel(
        node_path=node_path,
        dqr_path=dqr_path,
        edges_path=edges_path,
        output_path=output_path,
        filtered_edges_output_path=filter_edges_output_path,
        workers=workers,
        chunk_size=100_000,
    )


if __name__ == '__main__':
    construct_baseline()
