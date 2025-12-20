import csv
from pathlib import Path

import pytest

from tgrag.dataset.process_labels import (
    _load_domains,
    check_overlaps,
    merge_processed_labels,
    process_csv,
    process_goggle,
    process_unlabelled_csv,
)


@pytest.fixture
def no_check_processed_file(monkeypatch):
    monkeypatch.setattr(
        'tgrag.dataset.process_labels.check_processed_file',
        lambda path: None,
        raising=True,
    )


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    return tmp_path / 'input.csv'


@pytest.fixture
def output_csv(tmp_path: Path) -> Path:
    return tmp_path / 'output.csv'


def write_dict_csv(path: Path, fieldnames, rows):
    path.write_text('', encoding='utf-8')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_as_list(path: Path):
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_process_csv_basic_averaging_and_binarization(
    tmp_csv, output_csv, no_check_processed_file
):
    # two domains with multiple numeric labels
    write_dict_csv(
        tmp_csv,
        fieldnames=['domain', 'label'],
        rows=[
            {'domain': 'a.com', 'label': '1'},
            {'domain': 'a.com', 'label': '0'},
            {'domain': 'b.com', 'label': '0.2'},
            {'domain': 'b.com', 'label': '0.7'},
            {'domain': '', 'label': '1'},  # ignored (empty domain)
        ],
    )

    process_csv(
        input_csv=tmp_csv,
        output_csv=output_csv,
        is_url=False,
        domain_col='domain',
        label_col='label',
        inverse=False,
    )

    rows = read_csv_as_list(output_csv)

    # Expect:
    # a.com: (1 + 0) / 2 = 0.5 -> 1
    # b.com: (0.2 + 0.7) / 2 = 0.45 -> 0
    assert rows == [
        {'domain': 'a.com', 'label': '1'},
        {'domain': 'b.com', 'label': '0'},
    ]


def test_process_csv_inverse_flag(tmp_csv, output_csv, no_check_processed_file):
    write_dict_csv(
        tmp_csv,
        fieldnames=['domain', 'label'],
        rows=[
            {
                'domain': 'a.com',
                'label': '1',
            },  # avg 1.0 -> normally 1, but inverse -> 0
            {
                'domain': 'b.com',
                'label': '0',
            },  # avg 0.0 -> normally 0, but inverse -> 1
        ],
    )

    process_csv(
        input_csv=tmp_csv,
        output_csv=output_csv,
        is_url=False,
        domain_col='domain',
        label_col='label',
        inverse=True,
    )

    rows = read_csv_as_list(output_csv)
    assert rows == [
        {'domain': 'a.com', 'label': '0'},
        {'domain': 'b.com', 'label': '1'},
    ]


def test_process_unlabelled_csv_legit(tmp_path, no_check_processed_file):
    input_txt = tmp_path / 'unlabelled.txt'
    output_csv = tmp_path / 'unlabelled_out.csv'

    # Mixed formatting, duplicates, comments in parentheses
    input_txt.write_text(
        '\n'.join(
            [
                'Example.com',
                'example.com (some note)',
                'foo.bar Some extra tokens',
                '',
                '"quoted.org"',
            ]
        ),
        encoding='utf-8',
    )

    process_unlabelled_csv(input_path=input_txt, output_csv=output_csv, is_legit=True)

    rows = read_csv_as_list(output_csv)
    # Expected domains: example.com, foo.bar, quoted.org (lowercased, deduped, sorted)
    assert rows == [
        {'domain': 'example.com', 'label': '1'},
        {'domain': 'foo.bar', 'label': '1'},
        {'domain': 'quoted.org', 'label': '1'},
    ]


def test_process_unlabelled_csv_phishing(tmp_path, no_check_processed_file):
    input_txt = tmp_path / 'phish.txt'
    output_csv = tmp_path / 'phish_out.csv'

    input_txt.write_text('bad.com\nworse.net\n', encoding='utf-8')

    process_unlabelled_csv(input_path=input_txt, output_csv=output_csv, is_legit=False)

    rows = read_csv_as_list(output_csv)
    assert rows == [
        {'domain': 'bad.com', 'label': '0'},
        {'domain': 'worse.net', 'label': '0'},
    ]


def test_process_goggle_basic(tmp_path, no_check_processed_file):
    goggle_path = tmp_path / 'rules.goggle'
    output_csv = tmp_path / 'goggle_out.csv'

    # Mix of boost, discard, downrank, comments, and lines without site=
    goggle_path.write_text(
        '\n'.join(
            [
                '! comment line',
                '',
                '$boost=2,site=good.com,reason=something',
                '$discard,site=bad.com',
                '$downrank,site=meh.com',  # ignored
                '$boost=2,reason=nothing',  # no site= -> ignored
                '$discard,site=other.org',
            ]
        ),
        encoding='utf-8',
    )

    process_goggle(goggle_path=goggle_path, output_csv=output_csv)

    rows = read_csv_as_list(output_csv)
    # Order should respect the order of valid rules encountered
    assert rows == [
        {'domain': 'good.com', 'label': '1'},
        {'domain': 'bad.com', 'label': '0'},
        {'domain': 'other.org', 'label': '0'},
    ]


def test_merge_processed_labels_merges_and_binarizes(tmp_path, no_check_processed_file):
    processed_dir = tmp_path
    output_csv = processed_dir / 'labels.csv'

    # first partial csv
    csv1 = processed_dir / 'source1.csv'
    write_dict_csv(
        csv1,
        fieldnames=['domain', 'label'],
        rows=[
            {'domain': 'www.a.com', 'label': '1'},
            {'domain': 'b.com', 'label': '0'},
        ],
    )

    # second partial csv
    csv2 = processed_dir / 'source2.csv'
    write_dict_csv(
        csv2,
        fieldnames=['domain', 'label'],
        rows=[
            {
                'domain': 'a.com',
                'label': '0',
            },  # combined with 1 from source1 -> avg 0.5 -> 1
            {'domain': 'b.com', 'label': '1'},  # avg 0.5 -> 1
            {'domain': 'c.com', 'label': '0'},  # only 0 -> 0
        ],
    )

    # Create an existing output_csv-like file to ensure it's ignored
    output_csv.write_text('domain,label\nignore.com,1\n', encoding='utf-8')

    merge_processed_labels(processed_dir=processed_dir, output_csv=output_csv)

    rows = read_csv_as_list(output_csv)
    # Domains are sorted by name in merge_processed_labels
    # a.com: (1 + 0) / 2 = 0.5 -> 1
    # b.com: (0 + 1) / 2 = 0.5 -> 1
    # c.com: 0 -> 0
    assert rows == [
        {'domain': 'a.com', 'label': '1'},
        {'domain': 'b.com', 'label': '1'},
        {'domain': 'c.com', 'label': '0'},
    ]


def test__load_domains_strips_www_and_ignores_empty(tmp_path):
    csv_path = tmp_path / 'domains.csv'
    write_dict_csv(
        csv_path,
        fieldnames=['domain', 'label'],
        rows=[
            {'domain': 'www.a.com', 'label': '1'},
            {'domain': 'b.com', 'label': '0'},
            {'domain': '', 'label': '1'},
            {'domain': None, 'label': '1'},  # type: ignore
        ],
    )

    domains = _load_domains(csv_path)
    assert domains == {'a.com', 'b.com'}


def test_check_overlaps_uses_load_and_prints_counts(tmp_path, capsys, monkeypatch):
    strong_path = tmp_path / 'strong.csv'
    weak_path = tmp_path / 'weak.csv'

    write_dict_csv(
        strong_path,
        fieldnames=['domain', 'pc1'],
        rows=[
            {'domain': 'a.com', 'pc1': '0.1'},
            {'domain': 'b.com', 'pc1': '0.2'},
        ],
    )

    write_dict_csv(
        weak_path,
        fieldnames=['domain', 'label'],
        rows=[
            {'domain': 'b.com', 'label': '1'},
            {'domain': 'c.com', 'label': '0'},
        ],
    )

    check_overlaps(strong_labels=strong_path, weak_labels=weak_path)

    captured = capsys.readouterr().out.strip().splitlines()
    # Expected sets:
    # strong = {a,b}
    # weak   = {b,c}
    # overlap = {b}
    # union   = {a,b,c}
    assert captured[0].startswith('# strong: 2')
    assert captured[1].startswith('# weak: 2')
    assert captured[2].startswith('# overlap: 1')
    assert captured[3].startswith('# union: 3')
