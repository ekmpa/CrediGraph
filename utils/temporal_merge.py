import gzip
import os
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd

# this scripts merges multiple CC-MAIN slices into a temporal graph
# and allows for continual addition of new slices


class TemporalGraphMerger:
    """Merges multiple slices into a temporal graph (both edges and nodes are temporal).
    Then saves the graph (CSV) and can continually add slices to it.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir: str = output_dir
        self.edges: List[Tuple[int, int, int]] = []  # (src, dst, time_id)
        self.domain_to_node: Dict[str, int] = {}  # domain → node_id
        self.slice_node_sets: Dict[str, Set[int]] = {}  # slice_id → set of node_ids
        self.next_node_id: int = 0
        self.time_ids_seen: Set[int] = set()
        self._last_overlap: Optional[int] = None
        self._load_existing()

    def _load_existing(self) -> None:
        """Reconstruct graph from existing CSVs."""
        edges_path = os.path.join(self.output_dir, 'temporal_edges.csv')
        nodes_path = os.path.join(self.output_dir, 'temporal_nodes.csv')

        if os.path.exists(edges_path) and os.path.exists(nodes_path):
            df_edges = pd.read_csv(edges_path)
            df_nodes = pd.read_csv(nodes_path)

            self.edges = list(df_edges.itertuples(index=False, name=None))
            self.domain_to_node = {
                row['domain']: row['node_id'] for _, row in df_nodes.iterrows()
            }
            self.next_node_id = max(self.domain_to_node.values(), default=-1) + 1
            self.time_ids_seen = set(df_edges['time_id'])
            print(
                f'Loaded existing graph with {len(self.domain_to_node)} nodes and {len(self.edges)} edges'
            )

    def _normalize_domain(self, raw: str) -> str:
        """Normalize domain strings for consistency across slices."""
        raw = raw.strip().lower()
        if '://' in raw:
            raw = urlparse(raw).hostname or raw
        if raw.startswith('www.'):
            raw = raw[4:]
        if ':' in raw:
            raw = raw.split(':')[0]
        if raw.endswith('.'):
            raw = raw[:-1]
        return raw

    def _load_vertices(self, filepath: str) -> List[str]:
        """Helper to extract and load vertices from vertices.txt.gz."""
        domains = []
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                norm = self._normalize_domain(parts[1])
                domains.append(norm)
        return domains

    def _load_edges(self, filepath: str) -> List[Tuple[int, int]]:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            result: List[Tuple[int, int]] = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    result.append((int(parts[0]), int(parts[1])))
            return result

    def _slice_to_time_id(self, slice_id: str) -> int:
        """Yield timestamp from slice ID (current logic: YYYYMMDD)."""
        wat_dir = os.path.join('wat_files', slice_id)
        wat_files = sorted(glob(os.path.join(wat_dir, '*.wat.gz')))
        warc_date_re = re.compile(r'WARC-Date:\s*(\d{4})-(\d{2})-(\d{2})')

        for path in wat_files:
            try:
                with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = warc_date_re.search(line)
                        if match:
                            yyyy, mm, dd = match.groups()
                            return int(f'{yyyy}{mm}{dd}')
            except Exception as e:
                print(f'Warning: failed to parse {path}: {e}')
                continue

        raise ValueError(
            f'Could not extract scrape date from any WAT files in {wat_dir}'
        )

    def add_graph(self, vertices_path: str, edges_path: str, slice_id: str) -> None:
        """Add new slice to the existing temporal graph."""
        time_id = self._slice_to_time_id(slice_id)
        if time_id in self.time_ids_seen:
            print(f'Skipping slice {slice_id}: time_id {time_id} already exists.')
            return

        # snapshot current node IDs before mutation
        existing_node_ids = set(self.domain_to_node.values())

        # load vertices and edges using local -> global mapping
        domains = self._load_vertices(vertices_path)
        local_to_global = {}
        new_node_ids = set()

        for local_id, domain in enumerate(domains):
            if domain not in self.domain_to_node:
                self.domain_to_node[domain] = self.next_node_id
                self.next_node_id += 1

            node_id = self.domain_to_node[domain]
            local_to_global[local_id] = node_id
            new_node_ids.add(node_id)

        edges = self._load_edges(edges_path)
        for src_local, dst_local in edges:
            src_global = local_to_global.get(src_local)
            dst_global = local_to_global.get(dst_local)
            if src_global is not None and dst_global is not None:
                self.edges.append((src_global, dst_global, time_id))

        self.slice_node_sets[slice_id] = new_node_ids
        self.time_ids_seen.add(time_id)

        print(
            f'Added slice {slice_id} (timestamp {time_id}): {len(new_node_ids)} nodes, {len(edges)} edges'
        )

        # store overlap with pre-existing graph if this is the only slice being added now
        if len(self.slice_node_sets) == 1:
            self._last_overlap = len(existing_node_ids & new_node_ids)

    def save(self) -> None:
        """Save merged graph to CSV."""
        os.makedirs(self.output_dir, exist_ok=True)

        pd.DataFrame(self.edges, columns=['src', 'dst', 'time_id']).to_csv(
            os.path.join(self.output_dir, 'temporal_edges.csv'), index=False
        )

        df_nodes = pd.DataFrame(
            [
                {
                    'domain': domain,
                    'node_id': node_id,
                }
                for domain, node_id in self.domain_to_node.items()
            ]
        )
        df_nodes.to_csv(
            os.path.join(self.output_dir, 'temporal_nodes.csv'), index=False
        )

    def print_overlap(self) -> None:
        """Print overlap stats across all / added slices."""
        all_sets = list(self.slice_node_sets.values())

        if not all_sets:
            return

        if len(all_sets) == 1 and self._last_overlap is not None:
            print(f'Nodes in common with existing graph: {self._last_overlap}')
        else:
            common = set.intersection(*all_sets)
            print(f'Nodes in common across all slices: {len(common)}')


def main(slices: List[str]) -> None:
    base_path = 'external/cc-webgraph'
    output_dir = os.path.join(base_path, 'temporal')

    merger = TemporalGraphMerger(output_dir)

    for slice_id in slices:
        folder = slice_id.replace('-', '_')
        vertices_path = os.path.join(base_path, folder, 'vertices.txt.gz')
        edges_path = os.path.join(base_path, folder, 'edges.txt.gz')

        if not (os.path.exists(vertices_path) and os.path.exists(edges_path)):
            print(f'Missing data for {slice_id}: Skipping')
            continue

        merger.add_graph(vertices_path, edges_path, slice_id)

    merger.save()
    merger.print_overlap()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ./merge_pipeline.sh CC-MAIN-YYYY-NN [CC-MAIN-YYYY-NN ...]')
        sys.exit(1)
    main(sys.argv[1:])
