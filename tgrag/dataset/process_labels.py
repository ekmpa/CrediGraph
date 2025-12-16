import csv
from pathlib import Path

from tgrag.utils.data_loading import check_processed_file

# make all binary labels standardized and have 0 [unreliable] - 1 [reliable] and 'domain', 'label' columns as csv


def process_wiki(goggle_path: Path, output_csv: Path) -> None:
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


def main() -> None:
    classification_dir = Path('./data/classification')
    regression_dir = Path('./data/regression')

    class_raw = classification_dir / 'raw'
    class_proc = classification_dir / 'processed'

    regression_dir / 'raw'
    regression_dir / 'processed'

    process_wiki(
        Path(f'{class_raw}/wikipedia-reliable-sources.goggle'),
        Path(f'{class_proc}/wikipedia.csv'),
    )

    check_processed_file(Path(f'{class_proc}/wikipedia.csv'))


if __name__ == '__main__':
    main()
