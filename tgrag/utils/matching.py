"""Domain matching logic,
used for label matching, and WET-WAT matching.
"""

import gzip
from urllib.parse import urlparse

import pandas as pd
import tldextract


def extract_domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname or ''
    except Exception:
        return ''


def flip_if_needed(domain: str) -> str:
    """More comprehensive function for flipping. Use this instead of flip (deals with composite cases etc)."""
    extract = tldextract.TLDExtract(include_psl_private_domains=True)

    original_parts = domain.split('.')

    # Try correct order
    correct = extract(domain)
    '.'.join(part for part in [correct.domain, correct.suffix] if part)

    # Try reversed domain
    reversed_domain = '.'.join(reversed(original_parts))
    reversed_extract = extract(reversed_domain)
    '.'.join(
        part for part in [reversed_extract.domain, reversed_extract.suffix] if part
    )

    # Heuristic: pick the version with a known suffix and longer domain part
    if reversed_extract.suffix and len(reversed_extract.domain) >= len(correct.domain):
        return reversed_domain
    else:
        return domain


def force_flip(domain: str) -> str:
    parts = domain.strip().split('.')
    return '.'.join(reversed(parts))


def extract_registered_domain(url: str) -> str | None:
    url = flip_if_needed(url)
    ext = tldextract.extract(url)
    return f'{ext.domain}.{ext.suffix}' if ext.domain and ext.suffix else None


def extract_graph_domains(filepath: str) -> pd.DataFrame:
    parsed = []
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            domain = extract_registered_domain(line)
            parsed.append((i, domain))

    return pd.DataFrame(parsed, columns=['nid', 'domain'])


def reverse_domain(domain: str) -> str:
    return '.'.join(domain.split('.')[::-1])


def matches_suffix_pattern(domain_dqr: str, domain: str) -> bool:
    """Assuming that the domain_dqr is already reversed in the lookup table."""
    return domain == domain_dqr or domain.startswith(domain_dqr + '.')
