import os
from typing import Dict, List

from tgrag.utils.path import get_root_dir


def get_full_dict() -> Dict[str, List[float]]:
    """Get a dict with pc1 and every other metric."""
    path = os.path.join(get_root_dir(), 'data', 'dqr', 'domain_ratings.csv')
    wanted_domains: Dict[str, List[float]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            wanted_domains[parts[0]] = [float(x) for x in parts[1:]]
    return wanted_domains
