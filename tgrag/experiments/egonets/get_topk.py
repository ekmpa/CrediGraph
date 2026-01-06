from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import sys
from pathlib import Path
from typing import Iterable

from tgrag.utils.matching import flip_if_needed

DEFAULT_CRAWL = 'CC-MAIN-2025-05'
DEFAULT_PC1_FILE = Path('../../../data/dqr/domain_pc1.csv')


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'scratch',
        type=Path,
        help='Scratch directory that contains crawl-data/<crawl>/output1/vertices.csv.gz',
    )
    p.add_argument(
        'k',
        nargs='?',
        type=int,
        default=50,
        help='K for top/bottom selection (default: 50)',
    )
    p.add_argument(
        '--crawl',
        type=str,
        default=DEFAULT_CRAWL,
        help=f'Common Crawl name (default: {DEFAULT_CRAWL})',
    )
    p.add_argument(
        '--pc1-file',
        type=Path,
        default=DEFAULT_PC1_FILE,
        help=f'CSV with columns including domain,pc1 (default: {DEFAULT_PC1_FILE})',
    )
    return p.parse_args(argv)


def norm_domain(raw: str) -> str:
    return flip_if_needed(raw.strip().lower())


def load_pc1_map(path: Path) -> dict[str, float]:
    pc1_map: dict[str, float] = {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'PC1 file has no header: {path}')

        # Be slightly defensive about column naming.
        domain_key = 'domain'
        pc1_key = 'pc1'
        if domain_key not in reader.fieldnames or pc1_key not in reader.fieldnames:
            raise ValueError(
                f'PC1 file must include columns {domain_key!r} and {pc1_key!r}. '
                f'Found: {reader.fieldnames}'
            )

        for row in reader:
            dom = norm_domain(row[domain_key])
            try:
                pc1 = float(row[pc1_key])
            except (TypeError, ValueError):
                continue
            pc1_map[dom] = pc1
    return pc1_map


def iter_vertices_rows(vertices_gz: Path) -> tuple[list[str], Iterable[list[str]]]:
    """Returns (header, row_iter). Each row is a list[str] from csv.reader."""
    f = gzip.open(vertices_gz, 'rt', newline='')
    reader = csv.reader(f)

    try:
        header = next(reader)
    except StopIteration as e:
        f.close()
        raise ValueError(f'Vertices file is empty: {vertices_gz}') from e

    def _rows() -> Iterable[list[str]]:
        try:
            for row in reader:
                yield row
        finally:
            f.close()

    return header, _rows()


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.k <= 0:
        raise ValueError(f'k must be positive; got {args.k}')

    vertices_gz = (
        args.scratch / 'crawl-data' / args.crawl / 'output1' / 'vertices.csv.gz'
    )
    if not vertices_gz.exists():
        raise FileNotFoundError(f'Vertices file not found: {vertices_gz}')

    if not args.pc1_file.exists():
        raise FileNotFoundError(f'PC1 file not found: {args.pc1_file}')

    pc1_map = load_pc1_map(args.pc1_file)

    header, rows_iter = iter_vertices_rows(vertices_gz)

    best_row_by_domain: dict[str, tuple[float, list[str]]] = {}

    for row in rows_iter:
        if not row:
            continue
        raw_dom = row[0]
        dom = norm_domain(raw_dom)
        pc1 = pc1_map.get(dom)
        if pc1 is None:
            continue
        best_row_by_domain.setdefault(dom, (pc1, row))

    top_heap: list[tuple[float, str, list[str]]] = []
    bottom_heap: list[tuple[float, str, list[str]]] = []  # stores (-pc1, dom, row)
    k = args.k

    for dom, (pc1, row) in best_row_by_domain.items():
        top_item = (pc1, dom, row)
        if len(top_heap) < k:
            heapq.heappush(top_heap, top_item)
        elif pc1 > top_heap[0][0]:
            heapq.heapreplace(top_heap, top_item)

        neg_item = (-pc1, dom, row)
        if len(bottom_heap) < k:
            heapq.heappush(bottom_heap, neg_item)
        elif -pc1 > bottom_heap[0][0]:
            heapq.heapreplace(bottom_heap, neg_item)

    topk = sorted(top_heap, key=lambda x: x[0], reverse=True)
    bottomk = sorted(
        bottom_heap, key=lambda x: x[0]
    )  # most negative first => lowest pc1

    out = csv.writer(sys.stdout, lineterminator='\n')

    sys.stdout.write('### TOP K ###\n')
    out.writerow(['domain', 'pc1', *header[1:]])
    for pc1, dom, row in topk:
        out.writerow([dom, f'{pc1}', *row[1:]])

    sys.stdout.write('\n### BOTTOM K ###\n')
    out.writerow(['domain', 'pc1', *header[1:]])
    for neg_pc1, dom, row in bottomk:
        pc1 = -neg_pc1
        out.writerow([dom, f'{pc1}', *row[1:]])

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
