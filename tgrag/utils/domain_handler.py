import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import tldextract

_extract = tldextract.TLDExtract(include_psl_private_domains=True)


def normalize_domain(d: str | None) -> str | None:
    """Normalize a domain string.

    Parameters:
        d : str or None
            Domain string.

    Returns:
        str or None
            Normalized domain string without "www." prefix.
    """
    if not d:
        return None
    d = d.lower().strip()
    return d[4:] if d.startswith('www.') else d


def flip_if_needed(domain: str) -> str:
    """Normalize a possibly flipped domain (e.g., 'co.uk.theregister') into the
    canonical 'domain.suffix' (e.g., 'theregister.co.uk').

    Parameters:
        domain : str
            Input domain string.

    Returns:
        str
            Normalized domain in "domain.suffix" form.
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
    """Look up a domain by exact canonical match (with normalization).

    Parameters:
        domain : str
            Domain to search for.
        dqr_domains : dict[str, list[float]]
            Mapping from known domain strings to associated metric lists.

    Returns:
        list[float] or None
            Associated metric list if found, otherwise None.
    """
    domain_name = flip_if_needed(domain)
    return dqr_domains.get(domain_name)


def reverse_domain(domain: str) -> str:
    """Reverse the label order of a domain string.

    Parameters:
        domain : str
            Domain string.

    Returns:
        str
            Domain with label order reversed.
    """
    return '.'.join(domain.split('.')[::-1])


def extract_domain(raw: str) -> str | None:
    """Extract and normalize a domain from a raw string or URL.

    The input may be a bare domain, a URL, or a malformed string. The function
    attempts to normalize the input and extract a valid domain if possible.

    Parameters:
        raw : str
            Raw input string.

    Returns:
        str or None
            Extracted domain string, or None if extraction fails.
    """
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
