"""Domain matching logic,
used for label matching, and WET-WAT matching.
"""

from typing import Dict, List, Optional

import tldextract


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


def reverse_domain(domain: str) -> str:
    return '.'.join(domain.split('.')[::-1])
