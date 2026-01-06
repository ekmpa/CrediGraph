import csv
from pathlib import Path

import pytest

from tgrag.utils.checkers import check_overlaps, check_processed_labels


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    return tmp_path / "file.csv"


def write_dict_csv(path: Path, fieldnames, rows):
    path.write_text("", encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_csv(path: Path, rows):
    path.write_text("", encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


def test_check_overlaps_basic(tmp_path: Path, capsys):
    strong = tmp_path / "strong.csv"
    weak = tmp_path / "weak.csv"

    write_dict_csv(
        strong,
        ["domain", "pc1"],
        [
            {"domain": "a.com", "pc1": "0.1"},
            {"domain": "b.com", "pc1": "0.2"},
        ],
    )

    write_dict_csv(
        weak,
        ["domain", "label"],
        [
            {"domain": "b.com", "label": "1"},
            {"domain": "c.com", "label": "0"},
        ],
    )

    check_overlaps(strong, weak)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "# strong: 2",
        "# weak: 2",
        "# overlap: 1",
        "# union: 3",
    ]


def test_check_overlaps_empty_files(tmp_path: Path, capsys):
    strong = tmp_path / "strong.csv"
    weak = tmp_path / "weak.csv"

    write_dict_csv(strong, ["domain", "pc1"], [])
    write_dict_csv(weak, ["domain", "label"], [])

    check_overlaps(strong, weak)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "# strong: 0",
        "# weak: 0",
        "# overlap: 0",
        "# union: 0",
    ]


def test_check_overlaps_duplicate_domains(tmp_path: Path, capsys):
    strong = tmp_path / "strong.csv"
    weak = tmp_path / "weak.csv"

    write_dict_csv(
        strong,
        ["domain", "pc1"],
        [
            {"domain": "a.com", "pc1": "0.1"},
            {"domain": "a.com", "pc1": "0.2"},
        ],
    )

    write_dict_csv(
        weak,
        ["domain", "label"],
        [
            {"domain": "a.com", "label": "1"},
        ],
    )

    check_overlaps(strong, weak)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "# strong: 1",
        "# weak: 1",
        "# overlap: 1",
        "# union: 1",
    ]


def test_check_processed_labels_basic(tmp_path: Path, capsys):
    p = tmp_path / "processed.csv"

    write_raw_csv(
        p,
        [
            ["domain", "label"],
            ["a.com", "1"],
            ["b.com", "0"],
            ["c.com", "1"],
        ],
    )

    check_processed_labels(p)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "Processed: rows: 3",
        "Headers: ['domain', 'label']",
        "Label counts:",
        "  0: 1",
        "  1: 2",
    ]


def test_check_processed_labels_ignores_empty_rows(tmp_path: Path, capsys):
    p = tmp_path / "processed.csv"

    write_raw_csv(
        p,
        [
            ["domain", "label"],
            [],
            ["a.com", "1"],
            [],
            ["b.com", "0"],
        ],
    )

    check_processed_labels(p)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "Processed: rows: 2",
        "Headers: ['domain', 'label']",
        "Label counts:",
        "  0: 1",
        "  1: 1",
    ]


def test_check_processed_labels_only_header(tmp_path: Path, capsys):
    p = tmp_path / "processed.csv"

    write_raw_csv(p, [["domain", "label"]])

    check_processed_labels(p)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "Processed: rows: 0",
        "Headers: ['domain', 'label']",
        "Label counts:",
        "  0: 0",
        "  1: 0",
    ]


def test_check_processed_labels_unexpected_labels_ignored(tmp_path: Path, capsys):
    p = tmp_path / "processed.csv"

    write_raw_csv(
        p,
        [
            ["domain", "label"],
            ["a.com", "1"],
            ["b.com", "2"],  # ignored
            ["c.com", "-1"], # ignored
            ["d.com", "0"],
        ],
    )

    check_processed_labels(p)

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "Processed: rows: 4",
        "Headers: ['domain', 'label']",
        "Label counts:",
        "  0: 1",
        "  1: 1",
    ]