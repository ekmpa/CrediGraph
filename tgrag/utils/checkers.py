# Checker helpers,
# Incl. checkers for processed label files and graph files.

import csv
from pathlib import Path

from tgrag.utils.readers import load_domains

# For labels
# ----------


def check_overlaps(strong_labels: Path, weak_labels: Path) -> None:
    """Check and print overlaps between strong and weak label datasets.

    Parameters:
        strong_labels : pathlib.Path
            Path to CSV file containing strong labels,
            with columns 'domain', 'pc1' (to be changed to 'label' when we merge with other sources than DQR)
        weak_labels : pathlib.Path
            Path to CSV file containing weak labels,
            with columns 'domain' and 'label', label = 0 for phishing, label = 1 for legitimate

    Returns:
        None
    """
    strong = load_domains(strong_labels)
    weak = load_domains(weak_labels)

    overlap = strong & weak
    union = strong | weak

    print(f'# strong: {len(strong)}')
    print(f'# weak: {len(weak)}')
    print(f'# overlap: {len(overlap)}')
    print(f'# union: {len(union)}')


def check_processed_labels(processed: Path) -> None:
    """Inspect a processed CSV file and print basic statistics.

    Prints:
      - Total number of non-empty data rows
      - The header row (if present)
      - Counts of label 0 and label 1

    Parameters:
        processed : pathlib.Path
            Path to the processed CSV file.

    Returns: None
    """
    processed_count = 0
    label_counts = {0: 0, 1: 0}
    headers = None

    with processed.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        for row in reader:
            if not row:
                continue
            processed_count += 1
            label = int(row[1])
            if label in label_counts:
                label_counts[label] += 1

    print('Processed: rows:', processed_count)
    print('Headers:', headers)
    print('Label counts:')
    print('  0:', label_counts[0])
    print('  1:', label_counts[1])
