# Merger scripts
# Incl. mergers for subsets of the graphs, and for classic csvs for labels datasets.

import csv
import glob
import gzip
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Set, Tuple

from tgrag.utils.checkers import check_processed_labels
from tgrag.utils.readers import (
    collect_merged,
    line_reader,
    read_edge_file,
    read_reg_scores,
    read_vertex_file,
    read_weak_labels,
)
from tgrag.utils.writers import write_aggr_labelled

# For graphs
# ----------


def append_target_nodes(source_base: str, target_base_root: str) -> Dict[int, str]:
    r"""Aggregate vertex files from a source tree into a single gzip-compressed target file.

    Each vertex file is expected to contain tab-separated lines of the form:
        "<node_id>\\t<domain>"

    Parameters:
        source_base : str
            Base directory containing a `vertices/` subdirectory with `.txt.gz` files.
        target_base_root : str
            Root directory where the aggregated `vertices.txt.gz` file is written.

    Returns:
        dict[int, str]
            Mapping from node ID to domain name extracted from the source vertex files.
    """
    source_dir: str = os.path.join(source_base, 'vertices')
    target_path: str = os.path.join(target_base_root, 'vertices.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return {}

    combined_vertices: Set[str] = set()
    id_to_domain: Dict[int, str] = {}

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                combined_vertices.add(line)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(read_vertex_file, p) for p in matches]
        for fut in as_completed(futures):
            result = fut.result()
            for line in result:
                parts = line.split('\t')
                domain = parts[1]
                combined_vertices.add(domain)
                id_to_domain[int(parts[0])] = domain

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for vertex in combined_vertices:
            f.write(vertex + '\n')

    print(f'[INFO] Aggregated {len(combined_vertices)} vertices → {target_path}')
    return id_to_domain


def append_target_edges(
    source_base: str, target_base_root: str, id_to_domain: Dict[int, str]
) -> None:
    r"""Aggregate edge files from a source tree into a single gzip-compressed target file.

    Each edge file is expected to contain tab-separated integer node IDs:
        "<src_id>\\t<dst_id>"

    Parameters:
        source_base : str
            Base directory containing an `edges/` subdirectory with `.txt.gz` files.
        target_base_root : str
            Root directory where the aggregated `edges.txt.gz` file is written.
        id_to_domain : dict[int, str]
            Mapping from numeric node IDs to domain names.

    Returns:
        None

    """
    source_dir: str = os.path.join(source_base, 'edges')
    target_path: str = os.path.join(target_base_root, 'edges.txt.gz')
    matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))

    if not matches:
        print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
        return

    combined_edges: Set[Tuple[str, str]] = set()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(read_edge_file, p, id_to_domain) for p in matches]
        for fut in as_completed(futures):
            combined_edges.update(fut.result())

    if os.path.exists(target_path):
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t', 1)
                combined_edges.add((parts[0], parts[1]))

    with gzip.open(target_path, 'wt', encoding='utf-8') as f:
        for src, dst in combined_edges:
            f.write(f'{src}\t{dst}\n')

    print(f'[INFO] Aggregated {len(combined_edges)} edges → {target_path}')


def extract_all_domains(vertices: str, edges: str, out_txt: str) -> None:
    """Extract all domain names from vertex and edge files into a single text file.

    Parameters:
        vertices_gz : str
            Path to the gzip-compressed vertex file.
        edges_gz : str
            Path to the gzip-compressed edge file (tab-separated src and dst).
        out_txt : str
            Path to the output text file.

    Returns:
        None
    """
    with open(out_txt, 'w', encoding='utf-8', newline='') as out:
        for line in line_reader(vertices):
            dom = line.strip()
            if dom:
                out.write(dom + '\n')
        for line in line_reader(edges):
            if not line:
                continue
            try:
                src, dst = line.split('\t', 1)
            except ValueError:
                continue
            src = src.strip()
            dst = dst.strip()
            if src:
                out.write(src + '\n')
            if dst:
                out.write(dst + '\n')


# For labels
# ----------


def merge_processed_labels(processed_dir: Path, output_csv: Path) -> None:
    """Merge multiple processed label CSVs into a single aggregated output file.

    Parameters:
        processed_dir : pathlib.Path
            Directory containing processed CSV label files.
        output_csv : pathlib.Path
            Path where the merged CSV will be written.

    Returns:
        None
    """
    csv_paths = list(processed_dir.glob('*.csv'))
    domain_labels = collect_merged(csv_paths, output_csv)
    write_aggr_labelled(domain_labels, output_csv)
    check_processed_labels(output_csv)


def merge_reg_class(
    weak_labels_csv: Path,
    reg_csv: Path,
    output_csv: Path,
) -> None:
    """Merge weak classification labels and regression scores into a single CSV.

    Final output schema:
    domain, weak_label, reg_score
        - Many domains only have one of the two, then the other is None.

    Parameters:
        weak_labels_csv : pathlib.Path
            Path to CSV file containing weak labels.
        reg_csv : pathlib.Path
            Path to CSV file containing regression scores.
        output_csv : pathlib.Path
            Path where the merged CSV will be written.

    Returns:
        None
    """
    weak = read_weak_labels(weak_labels_csv)
    reg = read_reg_scores(reg_csv)

    all_domains = sorted(set(weak) | set(reg))

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'weak_label', 'reg_score'])

        for domain in all_domains:
            writer.writerow(
                [
                    domain,
                    weak.get(domain),
                    reg.get(domain),
                ]
            )
