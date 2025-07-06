"""Domain matching logic,
used for label matching, and WET-WAT matching.
"""

import gzip
import re
from urllib.parse import urlparse

import pandas as pd
import tldextract


def extract_domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname or ''
    except Exception:
        return ''


def flip(domain: str) -> str:
    parts = domain.split('.')
    # Only flip if it looks like TLD-domain pattern
    if len(parts) == 2 and parts[0] in {
        'com',
        'org',
        'net',
        'gov',
        'edu',
        'fr',
        'cn',
        'br',
        'au',
    }:
        return f'{parts[1]}.{parts[0]}'
    return domain


def extract_registered_domain(url: str) -> str | None:
    url = flip(url)
    ext = tldextract.extract(url)
    return f'{ext.domain}.{ext.suffix}' if ext.domain and ext.suffix else None


def extract_graph_domains(filepath: str) -> pd.DataFrame:
    parsed = []
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = re.sub(r'^\s*\d+\s+', '', line)  # remove node id

            if not line:
                continue

            domain = extract_registered_domain(line)
            parsed.append((i, domain))

    return pd.DataFrame(parsed, columns=['node_id', 'match_domain'])
