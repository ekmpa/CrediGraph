import gzip
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import tldextract


class Merger(ABC):
    """Abstract base class for merging two graphs.
    Used for temporal and article-level merging.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.edges: List[Tuple[int, int, int, str]] = []
        self.domain_to_node: Dict[str, Tuple[int, int]] = {}

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

    def _load_vertices(self, filepath: str) -> Tuple[List[str], List[int]]:
        """Helper to extract and load vertices from vertices.txt.gz."""
        domains = []
        node_ids = []
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                norm = self._normalize_domain(parts[1])
                if norm:
                    domains.append(norm)
                else:
                    continue
                node_ids.append(int(parts[0]))
        return domains, node_ids

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

        # Ensure all edges have 4 fields, defaulting edge_type to 'hyperlinks'
        processed_edges = [
            edge if len(edge) == 4 else (*edge, 'hyperlinks') for edge in self.edges
        ]

        edge_cols = ['src', 'dst', 'time_id', 'edge_type']
        pd.DataFrame(processed_edges, columns=edge_cols).to_csv(
            os.path.join(self.output_dir, 'temporal_edges.csv'), index=False
        )

        df_nodes = pd.DataFrame(
            [
                {'domain': domain, 'node_id': node_id, 'time_id': time_id}
                for domain, (node_id, time_id) in self.domain_to_node.items()
            ]
        )
        df_nodes.to_csv(
            os.path.join(self.output_dir, 'temporal_nodes.csv'), index=False
        )
