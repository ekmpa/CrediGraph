import gzip
import os
from typing import Dict, Tuple

import pandas as pd
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from tgrag.utils.matching import extract_registered_domain
from tgrag.utils.merger import Merger
from tgrag.utils.path import get_root_dir, get_wet_file_path


class ArticleMerger(Merger):
    """Merges a slice's WET files with the existing WAT-based graph.
    i.e, merge article-level to domain-level data for a given slice.
    """

    def __init__(self, output_dir: str, slice: str) -> None:
        super().__init__(output_dir)
        self.slice = slice
        self.matched_articles = 0
        self.unmatched_articles = 0
        self.article_nodes: Dict[
            str, Tuple[int, str, str]
        ] = {}  # url -> node_id, date, text
        self.next_node_id = (
            max((nid for nid, _ in self.domain_to_node.values()), default=0) + 1
        )

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
                domain_node_id, time_id = self.domain_to_node[domain]

                if wet_content['url'] not in self.article_nodes:
                    article_node_id = self.next_node_id
                    self.next_node_id += 1
                    self.article_nodes[wet_content['url']] = (
                        article_node_id,
                        wet_content['warc_date'],
                        wet_content['text'],
                    )
                else:
                    (article_node_id, _, _) = self.article_nodes[wet_content['url']]

                self.edges.append(
                    (domain_node_id, article_node_id, time_id, 'contains')
                )

            total_articles = self.matched_articles + self.unmatched_articles
            match_pct = (
                (self.matched_articles / total_articles * 100)
                if total_articles > 0
                else 0
            )

    def save(self) -> None:
        """Override save to handle both domain and article nodes."""
        os.makedirs(self.output_dir, exist_ok=True)

        processed_edges = [
            edge if len(edge) == 4 else (*edge, 'hyperlinks') for edge in self.edges
        ]
        pd.DataFrame(
            processed_edges, columns=['src', 'dst', 'time_id', 'edge_type']
        ).to_csv(os.path.join(self.output_dir, 'temporal_edges.csv'), index=False)

        domain_node_rows = [
            {'domain': domain, 'node_id': node_id, 'time_id': time_id}
            for domain, (node_id, time_id) in self.domain_to_node.items()
        ]

        article_node_rows = [
            {'domain': url, 'node_id': node_id, 'date': date, 'text': text}
            for url, (node_id, date, text) in self.article_nodes.items()
        ]

        df_nodes = pd.DataFrame(domain_node_rows + article_node_rows)
        df_nodes.to_csv(
            os.path.join(self.output_dir, 'temporal_nodes.csv'), index=False
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
