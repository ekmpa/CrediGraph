"""Domain matching logic,
used for label matching, and WET-WAT matching.
"""

import gzip
import re

import pandas as pd


def extract_graph_domains(filepath: str) -> pd.DataFrame:
    parsed = []
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip().lower()
            line = re.sub(r'^\s*\d+\s+', '', line)  # remove node id

            if not line:
                continue

            tokens = line.split('.')
            if len(tokens) >= 2:
                domain = tokens[1] + '.' + tokens[0]
            else:
                # this case shouldn't happen (fallback)
                domain = tokens[0]

            parsed.append((i, domain))

    return pd.DataFrame(parsed, columns=['node_id', 'match_domain'])
