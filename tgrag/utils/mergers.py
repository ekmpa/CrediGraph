import gzip
import json
import os
import re
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import tldextract
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from tgrag.utils.matching import extract_registered_domain
from tgrag.utils.path import get_root_dir, get_wet_file_path


class Merger(ABC):
    """Abstract base class for merging two graphs.
    Used for temporal and article-level merging.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.edges: List[Tuple[int, int, int]] = []  # src, dst, tid
        self.domain_to_node: Dict[
            str, Tuple[int, float | None, List[str]]
        ] = {}  # domain, (id, [texts])

    @abstractmethod
    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Merge method to be implemented by subclasses."""

    def _normalize_domain(self, url: str | None) -> str | None:
        """Extract and normalize domain from a URL or raw domain."""
        if not url:
            return None
        url = url.strip().lower()

        # Try to parse as a URL first
        parsed = urlparse(url)
        hostname = parsed.hostname or url

        ext = tldextract.extract(hostname)

        if ext.domain and ext.suffix:
            return f'{ext.suffix}.{ext.domain}'
        elif ext.domain and not ext.suffix:
            if hostname.count('.') == 1:  # if already a domain
                return hostname
            else:
                return ext.domain
        else:
            return hostname

    def _load_vertices(
        self, filepath: str
    ) -> Tuple[List[str], List[int], List[Optional[float]]]:
        """Helper to extract and load vertices from annotated vertices csv file."""
        vertices_df = pd.read_csv(filepath)
        node_ids = vertices_df['node_id'].tolist()
        domains = vertices_df['match_domain'].tolist()
        pc1_scores = vertices_df['pc1'].tolist()
        return domains, node_ids, pc1_scores

    def _load_edges(self, filepath: str) -> List[Tuple[int, int]]:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            result: List[Tuple[int, int]] = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    result.append((int(parts[0]), int(parts[1])))
            return result

    def save(self) -> None:
        """Save merged graph to CSV."""
        os.makedirs(self.output_dir, exist_ok=True)
        processed_edges = [edge for edge in self.edges]

        edge_cols = ['src', 'dst', 'tid']
        pd.DataFrame(processed_edges, columns=edge_cols).to_csv(
            os.path.join(self.output_dir, 'temporal_edges.csv'), index=False
        )

        df_nodes = pd.DataFrame(
            [
                {
                    'domain': domain,
                    'node_id': node_id,
                    'pc1': label,
                    'text': json.dumps(text),
                }
                for domain, (node_id, label, text) in self.domain_to_node.items()
            ]
        )
        df_nodes.to_csv(
            os.path.join(self.output_dir, 'temporal_nodes.csv'), index=False
        )

    def _deserialize_text(self, text_field: Any) -> List[str]:
        """Safe helper to parse the `text` column back to a list."""
        if text_field is None or pd.isna(text_field):
            return []
        try:
            return json.loads(text_field)
        except Exception:
            return []


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


class ArticleMerger(Merger):
    """Merges a slice's WET files with the existing WAT-based graph.
    i.e, merge article-level to domain-level data for a given slice.
    """

    def __init__(self, output_dir: str, slice: str) -> None:
        super().__init__(output_dir)
        self.slice = slice
        self.matched_articles = 0
        self.unmatched_articles = 0

    def merge(self) -> None:
        wet_path = get_wet_file_path(self.slice, str(get_root_dir()))

        with gzip.open(wet_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type != 'conversion':
                    continue

                wet_content = self._extract_wet_content(record)

                if not wet_content['url'] or not wet_content['text']:
                    continue

                domain = self._normalize_domain(
                    extract_registered_domain(wet_content['url'])
                )

                if domain not in self.domain_to_node:
                    self.unmatched_articles += 1
                    continue

                self.matched_articles += 1
                node_id, label, texts = self.domain_to_node[domain]
                texts.append(wet_content['text'])

            total_articles = self.matched_articles + self.unmatched_articles
            match_pct = (
                (self.matched_articles / total_articles * 100)
                if total_articles > 0
                else 0
            )

    def _extract_wet_content(self, record: ArcWarcRecord) -> dict:
        headers = record.rec_headers
        url = headers.get_header('WARC-Target-URI')
        warc_date = headers.get_header('WARC-Date')
        record_id = headers.get_header('WARC-Record-ID')
        content_type = headers.get_header('Content-Type')
        text = record.content_stream().read().decode('utf-8', errors='ignore')

        return {
            'url': url,
            'warc_date': warc_date,
            'record_id': record_id,
            'content_type': content_type,
            'text': text,
        }
