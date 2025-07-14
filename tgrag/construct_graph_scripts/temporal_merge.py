import gzip
import os
import re
from glob import glob
from typing import Dict, Optional, Set

import pandas as pd

from tgrag.utils.merger import Merger

# this scripts merges multiple CC-MAIN slices into a temporal graph
# and allows for continual addition of new slices


class TemporalGraphMerger(Merger):
    """Merges multiple slices into a temporal graph (both edges and nodes are temporal).
    Then saves the graph (CSV) and can continually add slices to it.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

        self.slice_node_sets: Dict[str, Set[int]] = {}  # slice_id → set of node_ids
        self.time_ids_seen: Set[int] = set()
        self._last_overlap: Optional[int] = None
        self._load_existing()

    def _load_existing(self) -> None:
        """Reconstruct graph from existing CSVs."""
        next_edges_path = os.path.join(self.output_dir, 'temporal_edges.csv')
        nodes_path = os.path.join(self.output_dir, 'temporal_nodes.csv')

        if os.path.exists(next_edges_path) and os.path.exists(nodes_path):
            try:
                df_edges = pd.read_csv(next_edges_path)
                df_nodes = pd.read_csv(nodes_path)
                # self.edges = list(df_edges.itertuples(index=False, name=None))
                self.edges = [
                    (row[0], row[1], row[2])
                    for row in df_edges.itertuples(index=False, name=None)
                ]  # should be able to go back to above with clean csvs. keeping this to ensure rn.
                self.domain_to_node = {
                    row['domain']: (
                        row['node_id'],
                        row['pc1'],
                        self._deserialize_text(row['text']),
                    )
                    for _, row in df_nodes.iterrows()
                }
                self.time_ids_seen = set(df_edges['tid'])
                print(
                    f'Loaded existing graph with {len(self.domain_to_node)} nodes and {len(self.edges)} edges'
                )
            except Exception as e:
                print(
                    f'Error occured in reading csv, they are likely empty. Error: {e}'
                )
                print('Continuing without loading existing CSVs.')

    def _slice_to_time_id(self, next_root_path: str, slice_id: str) -> int:
        """Yield timestamp from slice ID (current logic: YYYYMMDD)."""
        pattern = os.path.join(next_root_path, slice_id, 'segments', '*', 'wat')
        wat_dir = glob(pattern)[0]
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

    def merge(
        self,
        next_root_path: str,
        next_vertices_path: str,
        next_edges_path: str,
        slice_id: str,
    ) -> None:
        """Add new slice to the existing temporal graph."""
        tid = self._slice_to_time_id(next_root_path, slice_id)
        if tid in self.time_ids_seen:
            print(f'Skipping slice {slice_id}: tid {tid} already exists.')
            return

        # snapshot current node IDs before mutation
        existing_node_ids = {val[0] for val in self.domain_to_node.values()}

        # load vertices and edges using local -> global mapping
        domains, node_ids, labels = super()._load_vertices(next_vertices_path)
        new_node_ids = set()

        for local_id, domain in enumerate(domains):
            node_id = node_ids[local_id]
            label = labels[local_id]
            if domain not in self.domain_to_node:
                new_node_ids.add(node_id)
            self.domain_to_node[domain] = (node_id, label, [])  # tid, [])

        edges = super()._load_edges(next_edges_path)
        for src_local, dst_local in edges:
            self.edges.append((src_local, dst_local, tid))

        self.slice_node_sets[slice_id] = new_node_ids
        self.time_ids_seen.add(tid)

        print(
            f'Added slice {slice_id} (timestamp {tid}): {len(new_node_ids)} nodes, {len(edges)} edges'
        )

        # store overlap with pre-existing graph if this is the only slice being added now
        if len(self.slice_node_sets) == 1:
            self._last_overlap = len(existing_node_ids & new_node_ids)

        super().save()

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
