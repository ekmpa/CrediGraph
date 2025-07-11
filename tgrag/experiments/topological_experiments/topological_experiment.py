import csv
import gzip
from collections import Counter

from tgrag.utils.args import DataArguments


def run_topological_experiment(data_args: DataArguments, experiment: str) -> str:
    slice_id = data_args.slice_id
    node_file = data_args.node_file
    edge_file = data_args.edge_file

    print(f'Running topological experiment [{experiment}] on slice: {slice_id}')
    print(f'Node file: {node_file}')
    print(f'Edge file: {edge_file}')

    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()

    assert isinstance(
        edge_file, str
    ), 'Lists of edge files only valid for GNN experiments.'

    with gzip.open(edge_file, mode='rt', newline='') as gzfile:
        reader = csv.reader(gzfile, delimiter='\t')
        for row in reader:
            if len(row) != 2:
                continue
            src_id, dst_id = row[0].strip(), row[1].strip()
            out_degree[src_id] += 1
            in_degree[dst_id] += 1

    if experiment == 'IN_DEG':
        max_in = in_degree.most_common(1)
        result_str = (
            f'Slice: {slice_id}\n'
            f'Max in-degree: {max_in[0][1]} (Node ID: {max_in[0][0]})'
            if max_in
            else 'No edges found or file malformed.'
        )
    elif experiment == 'OUT_DEG':
        max_out = out_degree.most_common(1)
        result_str = (
            f'Slice: {slice_id}\n'
            f'Max out-degree: {max_out[0][1]} (Node ID: {max_out[0][0]})'
            if max_out
            else 'No edges found or file malformed.'
        )
    else:
        result_str = f'Unknown experiment type: {experiment}'

    print(result_str)
    return result_str
