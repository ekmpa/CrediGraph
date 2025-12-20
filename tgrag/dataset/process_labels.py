import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from tgrag.utils.data_loading import check_processed_file
from tgrag.utils.matching import extract_domain

# make all binary labels standardized and have 0 [unreliable] - 1 [reliable] and 'domain', 'label' columns as csv


def process_csv(
    input_csv: Path,
    output_csv: Path,
    is_url: bool,
    domain_col: str,
    label_col: str,
    inverse: bool = False,
    labels: Optional[List] = None,
) -> None:
    domain_labels = defaultdict(list)

    with input_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            domain = row.get(domain_col)
            label = row.get(label_col)

            if not domain or label is None:
                continue

            if is_url:
                domain = extract_domain(domain)

            score = label
            if labels is not None:
                if label == labels[0]:
                    score = 0
                elif label == labels[1]:
                    score = 1
                else:
                    continue

            try:
                domain_labels[domain].append(float(score))
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
            if inverse:
                binary_label = 1 - binary_label
            writer.writerow([domain, binary_label])

    check_processed_file(output_csv)


def process_unlabelled_csv(input_path: Path, output_csv: Path, is_legit: bool) -> None:
    label = 1 if is_legit else 0

    domains = set()

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue

            line = re.sub(r'\s*\(.*?\)\s*$', '', line)
            domain = line.split()[0].lower()

            if domain:
                domains.add(domain)

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain in sorted(domains):
            writer.writerow([domain, label])

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


def merge_processed_labels(processed_dir: Path, output_csv: Path) -> None:
    domain_labels = defaultdict(list)

    for csv_path in processed_dir.glob('*.csv'):
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

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])

        for domain, labels in sorted(domain_labels.items()):
            if not labels:
                continue

            avg_label = sum(labels) / len(labels)
            final_label = 1 if avg_label >= 0.5 else 0
            writer.writerow([domain, final_label])

    check_processed_file(output_csv)


def _load_domains(path: Path, domain_col: str = 'domain') -> set:
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


def check_overlaps(strong_labels: Path, weak_labels: Path) -> None:
    """Checks overlaps between used datasets.

    Assumption:
    - The file at `strong_labels` has a path with columns 'domain', 'pc1' (to be changed to 'label' when we merge with other sources than DQR)
    - The file at `weak_labels` has columns 'domain' and 'label', label = 0 for phishing, label = 1 for legitimate

    """
    strong = _load_domains(strong_labels)
    weak = _load_domains(weak_labels)

    overlap = strong & weak
    union = strong | weak

    print(f'# strong: {len(strong)}')
    print(f'# weak: {len(weak)}')
    print(f'# overlap: {len(overlap)}')
    print(f'# union: {len(union)}')


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
        is_url=True,
        domain_col='URL',
        label_col='ClassLabel',
        inverse=False,
    )

    print('======= PhishDataset ========')
    process_csv(
        Path(f'{class_raw}/data_imbal.csv'),
        Path(f'{class_proc}/phish-dataset.csv'),
        is_url=True,
        domain_col='URLs',
        label_col='\ufeffLabels',
        inverse=True,
    )

    print('======= Nelez ========')
    process_unlabelled_csv(
        Path(f'{class_raw}/dezinformacni_weby (2).csv'),
        Path(f'{class_proc}/nelez.csv'),
        is_legit=False,
    )

    print('======= wiki ========')
    process_goggle(
        Path(f'{class_raw}/wikipedia-reliable-sources.goggle'),
        Path(f'{class_proc}/wikipedia.csv'),
    )

    print('======= URL-Phish ========')
    process_csv(
        Path(f'{class_raw}/Dataset.csv'),
        Path(f'{class_proc}/url-phish.csv'),
        is_url=True,
        domain_col='url',
        label_col='label',
        inverse=True,
    )

    print('======== Phish&Legit =======')
    process_csv(
        Path(f'{class_raw}/new_data_urls.csv'),
        Path(f'{class_proc}/phish-and-legit.csv'),
        is_url=True,
        domain_col='url',
        label_col='status',
        inverse=False,
    )

    print('======== Misinformation domains =========')
    process_csv(
        Path(f'{class_raw}/domain_list_clean.csv'),
        Path(f'{class_proc}/misinfo-domains.csv'),
        is_url=False,
        domain_col='url',
        label_col='type',
        inverse=False,
        labels=['unreliable', 'reliable'],
    )

    print('======== Merging final labels =========')
    merge_processed_labels(
        class_proc,
        Path(f'{class_proc}/labels.csv'),
    )

    check_overlaps(
        Path('./data/dqr/domain_pc1.csv'),
        Path(f'{class_proc}/labels.csv'),
    )


if __name__ == '__main__':
    main()
