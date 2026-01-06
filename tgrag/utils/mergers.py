# Merger scripts
# Incl. mergers for subsets of the graphs, and for classic csvs for labels datasets.

import csv
from pathlib import Path

from tgrag.utils.checkers import check_processed_labels
from tgrag.utils.readers import (
    collect_merged,
    read_reg_scores,
    read_weak_labels,
)
from tgrag.utils.writers import write_aggr_labelled

# For labels
# ----------


def merge_processed_labels(processed_dir: Path, output_csv: Path) -> None:
    """Merge multiple processed label CSVs into a single aggregated output file.

    Parameters:
        processed_dir : pathlib.Path
            Directory containing processed CSV label files.
        output_csv : pathlib.Path
            Path where the merged CSV will be written.

    Returns:
        None
    """
    csv_paths = list(processed_dir.glob('*.csv'))
    domain_labels = collect_merged(csv_paths, output_csv)
    write_aggr_labelled(domain_labels, output_csv)
    check_processed_labels(output_csv)


def merge_reg_class(
    weak_labels_csv: Path,
    reg_csv: Path,
    output_csv: Path,
) -> None:
    """Merge weak classification labels and regression scores into a single CSV.

    Final output schema:
    domain, weak_label, reg_score
        - Many domains only have one of the two, then the other is None.

    Parameters:
        weak_labels_csv : pathlib.Path
            Path to CSV file containing weak labels.
        reg_csv : pathlib.Path
            Path to CSV file containing regression scores.
        output_csv : pathlib.Path
            Path where the merged CSV will be written.

    Returns:
        None
    """
    weak = read_weak_labels(weak_labels_csv)
    reg = read_reg_scores(reg_csv)

    all_domains = sorted(set(weak) | set(reg))

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'weak_label', 'reg_score'])

        for domain in all_domains:
            writer.writerow(
                [
                    domain,
                    weak.get(domain),
                    reg.get(domain),
                ]
            )
