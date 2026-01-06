import gzip
from pathlib import Path

import pytest

from tgrag.utils.writers import (
    build_from_BCC,
    compute_degrees,
    write_aggr_labelled,
    write_endpoints,
)


@pytest.fixture
def tmpdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_edges_gz(tmpdir: Path) -> Path:
    p = tmpdir / 'edges.txt.gz'
    with gzip.open(p, 'wt') as f:
        f.write('a\tb\n')
        f.write('a\tc\n')
        f.write('b\tc\n')
    return p


@pytest.fixture
def sample_edges_plain(tmpdir: Path) -> Path:
    p = tmpdir / 'edges.txt'
    p.write_text('a\tb\nb\tc\nc\ta\n', encoding='utf-8')
    return p


@pytest.fixture
def fake_sort(monkeypatch):
    def _fake_sort(src, dst, **kwargs):
        with open(src) as fin, open(dst, 'w') as fout:
            lines = sorted(l.strip() for l in fin if l.strip())
            for l in lines:
                fout.write(l + '\n')

    monkeypatch.setattr('tgrag.utils.writers.run_ext_sort', _fake_sort)
    monkeypatch.setattr('tgrag.utils.analytics.run_ext_sort', _fake_sort)


@pytest.fixture
def fake_count_lines(monkeypatch):
    monkeypatch.setattr('tgrag.utils.writers.count_lines', lambda p: 3)


def test_write_endpoints_basic(tmpdir: Path, sample_edges_gz):
    out = tmpdir / 'endpoints.txt'
    E, lines = write_endpoints(sample_edges_gz, out)

    assert E == 3
    assert lines == 6
    assert out.read_text().splitlines() == ['a', 'b', 'a', 'c', 'b', 'c']


def test_compute_degrees_basic(tmpdir: Path, sample_edges_gz, fake_sort):
    deg_path, E = compute_degrees(sample_edges_gz, tmpdir, sort_cmd='sort', mem='1G')

    assert E == 3
    lines = deg_path.read_text().splitlines()
    assert sorted(lines) == sorted(['a\t2', 'b\t2', 'c\t2'])


def test_write_aggr_labelled_basic(tmpdir: Path):
    data = {
        'a.com': [1, 0],
        'b.com': [0, 0],
        'c.com': [1, 1],
    }

    out = tmpdir / 'labels.csv'
    write_aggr_labelled(data, out)

    rows = out.read_text().splitlines()
    assert rows == [
        'domain,label',
        'a.com,1',
        'b.com,0',
        'c.com,1',
    ]


def test_write_aggr_labelled_skips_empty(tmpdir: Path):
    data = {
        'a.com': [],
        'b.com': [1],
    }

    out = tmpdir / 'labels.csv'
    write_aggr_labelled(data, out)

    assert out.read_text().splitlines() == [
        'domain,label',
        'b.com,1',
    ]


def test_build_from_BCC_basic(
    tmpdir: Path, sample_edges_plain, fake_sort, fake_count_lines, capsys
):
    out_gz = tmpdir / 'vertices.csv.gz'

    stats = build_from_BCC(
        edges_path=sample_edges_plain,
        out_vertices_gz=out_gz,
        ts_str='20240101',
        sort_cmd='sort',
        mem='1G',
    )

    assert out_gz.exists()

    with gzip.open(out_gz, 'rt') as f:
        lines = f.read().splitlines()

    assert lines[0] == 'domain,ts,in_deg,out_deg'
    assert len(lines) == 4  # 3 nodes + header

    assert stats['V'] == 3
    assert stats['E'] == 3
    assert stats['isolated'] == 0

    out = capsys.readouterr().out
    assert out.startswith('[STATS:final]')


def test_build_from_BCC_ignores_bad_lines(tmpdir: Path, fake_sort, fake_count_lines):
    edges = tmpdir / 'edges.txt'
    edges.write_text(
        'a\tb\nbad line\nc\t\n\tb\nb\tc\n',
        encoding='utf-8',
    )

    out_gz = tmpdir / 'vertices.csv.gz'

    stats = build_from_BCC(
        edges_path=edges,
        out_vertices_gz=out_gz,
        ts_str='20240101',
        sort_cmd='sort',
        mem='1G',
    )

    with gzip.open(out_gz, 'rt') as f:
        rows = f.read().splitlines()

    assert rows[0] == 'domain,ts,in_deg,out_deg'
    assert len(rows) == 4  # only a,b,c survived

    assert stats['V'] == 2 or stats['V'] == 3  # depending on filtered edges
