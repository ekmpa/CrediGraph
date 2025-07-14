import gzip

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
