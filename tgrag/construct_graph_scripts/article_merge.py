import argparse
import os

from tgrag.utils.article_merger import ArticleMerger
from tgrag.utils.path import get_crawl_data_path, get_root_dir


def merge(slice_id: str) -> None:
    base_path = get_root_dir()
    crawl_path = get_crawl_data_path(base_path)
    output_dir = os.path.join(crawl_path, slice_id, 'article_level')

    os.makedirs(output_dir, exist_ok=True)

    # Init merger
    merger = ArticleMerger(output_dir=output_dir, slice=slice_id)

    # Load domain-level graph
    vertices_path = os.path.join(
        crawl_path, slice_id, 'output_text_dir', 'vertices.txt.gz'
    )
    edges_path = os.path.join(crawl_path, slice_id, 'output_text_dir', 'edges.txt.gz')

    if not os.path.exists(vertices_path):
        print(f'Missing vertices: {vertices_path}')
        return
    if not os.path.exists(edges_path):
        print(f'Missing edges: {edges_path}')
        return

    # Load vertices and edges into merger
    domains, node_ids = merger._load_vertices(vertices_path)
    time_id = int(slice_id.split('-')[-1])  # crude method for example
    merger.domain_to_node = {
        domain: (nid, time_id) for domain, nid in zip(domains, node_ids)
    }
    merger.edges.extend(
        [
            (src, dst, time_id, 'hyperlinks')
            for src, dst in merger._load_edges(edges_path)
        ]
    )

    merger.merge()
    merger.save()

    # Count edges by type
    edge_type_counts = {'hyperlinks': 0, 'contains': 0}
    for _, _, _, edge_type in merger.edges:
        if edge_type in edge_type_counts:
            edge_type_counts[edge_type] += 1

    print(f'Done. Merged graph saved to {output_dir}')
    # print(f"Edge summary:")
    # print(f"  Hyperlinks: {edge_type_counts['hyperlinks']}")
    # print(f"  Contains:   {edge_type_counts['contains']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge WET articles with domain-level graph.'
    )
    parser.add_argument(
        '--slice', required=True, help='CC-MAIN time slice (e.g. CC-MAIN-2017-13)'
    )
    args = parser.parse_args()

    merge(args.slice)
