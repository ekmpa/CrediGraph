import csv
from collections import defaultdict
from pathlib import Path

from tgrag.utils.data_loading import check_processed_file
from tgrag.utils.matching import extract_domain

# make all binary labels standardized and have 0 [unreliable] - 1 [reliable] and 'domain', 'label' columns as csv


def process_csv(input_csv: Path, output_csv: Path) -> None:
    domain_labels = defaultdict(list)

    with input_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            url = row.get('URL')
            label = row.get('ClassLabel')

            if url is None or label is None:
                continue

            domain = extract_domain(url)
            if domain is None:
                continue

            try:
                domain_labels[domain].append(float(label))
            except ValueError:
                continue

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in domain_labels.items():
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            binary_label = 1 if avg_label >= 0.5 else 0
            writer.writerow([domain, binary_label])

    check_processed_file(output_csv)


def process_goggle(goggle_path: Path, output_csv: Path) -> None:
    rows = []

    with goggle_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            if line.startswith('$boost=2'):
                label = 1
            elif line.startswith('$discard'):
                label = 0
            else:
                # ignore $downrank
                continue

            # extract domain
            parts = line.split(',')
            site_part = next((p for p in parts if p.startswith('site=')), None)
            if site_part is None:
                continue

            domain = site_part.split('=', 1)[1]
            rows.append((domain, label))

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        writer.writerows(rows)

    check_processed_file(output_csv)


def main() -> None:
    classification_dir = Path('./data/classification')
    regression_dir = Path('./data/regression')

    class_raw = classification_dir / 'raw'
    class_proc = classification_dir / 'processed'

    regression_dir / 'raw'
    regression_dir / 'processed'

    print('======= LegitPhish ========')
    process_csv(
        Path(f'{class_raw}/url_features_extracted1.csv'),
        Path(f'{class_proc}/legit-phish.csv'),
    )

    print('======= wiki ========')
    process_goggle(
        Path(f'{class_raw}/wikipedia-reliable-sources.goggle'),
        Path(f'{class_proc}/wikipedia.csv'),
    )


if __name__ == '__main__':
    main()
