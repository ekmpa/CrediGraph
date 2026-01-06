# File writers.
# Include writers and loaders used for graphs at various steps and label datasets.

import csv
from pathlib import Path

# For domain labels
# ------------------


def write_aggr_labelled(
    domain_labels: dict[str, list[float]], output_csv: Path
) -> None:
    """Write merged domain labels to a CSV file.

    For each domain, compute the average of its associated labels and assign a
    final binary label: 1 if the average is at least 0.5, otherwise 0. The output
    CSV contains one row per domain with columns "domain" and "label".

    Parameters:
        domain_labels : dict
            Mapping from domain to a list of numeric labels.
        output_csv : pathlib.Path
            Path to the output CSV file.

    Returns:
        None
    """
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in sorted(domain_labels.items()):
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            final_label = 1 if avg_label >= 0.5 else 0
            writer.writerow([domain, final_label])
