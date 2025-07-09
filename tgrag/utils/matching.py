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
    extract = tldextract.TLDExtract(include_psl_private_domains=True)
    extract.update()
    parts = domain.split('.')
    if (
        len(parts) == 2 and extract(parts[0]).suffix != ''
    ):  # Only flip if it looks like TLD-domain pattern
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


def reverse_domain(domain: str) -> str:
    return '.'.join(domain.split('.')[::-1])


def matches_suffix_pattern(domain_dqr: str, domain: str) -> bool:
    """Assuming that the domain_dqr is already reversed in the lookup table."""
    return domain == domain_dqr or domain.startswith(domain_dqr + '.')
