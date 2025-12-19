"""Domain matching logic,
used for label matching, and WET-WAT matching.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import tldextract

_extract = tldextract.TLDExtract(include_psl_private_domains=True)


def flip_if_needed(domain: str) -> str:
    """Normalize a possibly flipped domain (e.g., 'co.uk.theregister') into the
    canonical 'domain.suffix' (e.g., 'theregister.co.uk').

    Strategy: try all cyclic rotations of labels; pick the parse with
    the longest PSL suffix (# of labels), then longest domain label.
    """
    if not domain:
        return domain

    labels = [p for p in domain.strip('.').lower().split('.') if p]
    if not labels:
        return domain

    best = None  # (suffix_label_count, domain_len, normalized_str)

    n = len(labels)
    for r in range(n):
        # rotation: move the last r labels to the front
        rotated = labels[-r:] + labels[:-r] if r else labels
        rotated_str = '.'.join(rotated)

        ext = _extract(rotated_str)
        if not ext.suffix or not ext.domain:
            continue

        suffix_labels = ext.suffix.count('.') + 1  # e.g., 'co.uk' -> 2
        dom_len = len(ext.domain)
        normalized = f'{ext.domain}.{ext.suffix}'

        cand = (suffix_labels, dom_len, normalized)
        if (best is None) or (cand > best):
            best = cand

    # Fall back: parse the input as-is if no rotation produced a valid suffix
    if best is None:
        ext = _extract('.'.join(labels))
        if ext.suffix and ext.domain:
            return f'{ext.domain}.{ext.suffix}'
        return '.'.join(labels)

    return best[2]


def lookup(domain: str, dqr_domains: Dict[str, List[float]]) -> Optional[List[float]]:
    """Look up domain in dqr_domains, return associated data if found."""
    domain_parts = domain.split('.')
    for key, value in dqr_domains.items():
        key_parts = key.split('.')
        if (
            len(key_parts) >= 2
            and key_parts[0] in domain_parts
            and key_parts[1] in domain_parts
        ):
            return value
    return None


def lookup_exact(
    domain: str, dqr_domains: Dict[str, List[float]]
) -> Optional[List[float]]:
    domain_name = flip_if_needed(domain)
    return dqr_domains.get(domain_name)


def reverse_domain(domain: str) -> str:
    return '.'.join(domain.split('.')[::-1])


def extract_domain(raw: str) -> str | None:
    if not raw:
        return None

    raw = raw.strip().strip('\'"')
    raw = raw.replace('&amp;', '&')

    # if no scheme, add one so urlparse works
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw):
        raw = 'http://' + raw

    try:
        parsed = urlparse(raw)
        domain = parsed.netloc.lower()

        if not domain:
            return None

        domain = domain.split(':')[0]
        return domain

    except Exception:
        return None
