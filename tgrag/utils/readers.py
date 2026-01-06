# Reader helpers
# Incl. readers for graphs at various steps and label datasets.

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# For labels
# ----------


def load_domains(path: Path, domain_col: str = 'domain') -> set[str]:
    """Load and normalize domain names from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to CSV file containing domain values.
        domain_col : str, optional
            Name of the column containing domain strings.

    Returns:
        set[str]
            Set of normalized domain strings.
    """
    domains = set()
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get(domain_col)
            if not d:
                continue
            if d.startswith('www.'):
                d = d[4:]
            domains.add(d)
    return domains


def read_weak_labels(path: Path) -> dict[str, int]:
    """Read weak (binary) labels from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to a CSV file with columns "domain" and "label".

    Returns:
        dict[str, int]
            Mapping from domain to integer label.
    """
    weak = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get('domain')
            l = row.get('label')
            if not d or l is None:
                continue
            if d.startswith('www.'):
                d = d[4:]
            weak[d] = int(l)
    return weak


def read_reg_scores(path: Path, score_col: str = 'pc1') -> dict[str, float]:
    """Read regression scores from a CSV file.

    Parameters:
        path : pathlib.Path
            Path to a CSV file containing domain scores.
        score_col : str, optional
            Name of the column containing the regression score.

    Returns:
        dict[str, float]
            Mapping from domain to floating-point regression score.
    """
    reg = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get('domain')
            s = row.get(score_col)
            if not d or s is None:
                continue
            if d.startswith('www.'):
                d = d[4:]
            try:
                reg[d] = float(s)
            except ValueError:
                continue
    return reg


def collect_merged(paths: Iterable[Path], output_csv: Path) -> dict[str, list[float]]:
    """Collect and aggregate domain labels from multiple CSV files that have labelled domains.

    Parameters:
        paths : iterable of pathlib.Path
            Input CSV file paths containing at least columns "domain" and "label".
        output_csv : pathlib.Path
            Path to the output CSV file (excluded from reading).

    Returns:
        dict
            Mapping from domain string to a list of numeric label values.
    """
    domain_labels = defaultdict(list)

    for csv_path in paths:
        if csv_path.name == output_csv.name:
            continue

        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                domain = row.get('domain')
                if type(domain) == str and domain.startswith('www.'):
                    domain = domain[4:]
                label = row.get('label')

                if not domain or label is None:
                    continue

                try:
                    domain_labels[domain].append(float(label))
                except ValueError:
                    continue

    return domain_labels
