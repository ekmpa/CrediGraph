import gzip
from pathlib import Path

import pytest

from tgrag.utils.analytics import (
    compute_density,
    count_lines,
    count_sorted_keys,
    stats,
)
from tgrag.utils.writers import compute_degrees


@pytest.fixture
def tmpdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_edges(tmpdir: Path) -> Path:
    p = tmpdir / 'edges.txt.gz'
    with gzip.open(p, 'wt') as f:
        f.write('a\tb\n')
        f.write('a\tc\n')
        f.write('b\tc\n')
    return p


@pytest.fixture
def sample_vertices(tmpdir: Path) -> Path:
    p = tmpdir / 'verts.txt.gz'
    with gzip.open(p, 'wt') as f:
        f.write('a\nb\nc\n')
    return p


@pytest.fixture
def fake_sort(monkeypatch):
    def _fake_sort(src, dst, **kwargs):
        with open(src) as fin, open(dst, 'w') as fout:
            lines = sorted(l.strip() for l in fin if l.strip())
            for l in dict.fromkeys(lines):  # preserve unique if needed
                fout.write(l + '\n')

    monkeypatch.setattr('tgrag.utils.analytics.run_ext_sort', _fake_sort)


@pytest.fixture
def fake_run(monkeypatch):
    class FakeOut:
        def __init__(self, stdout):
            self.stdout = stdout

    def _fake_run(cmd):
        if cmd[0] == 'wc':
            with open(cmd[-1]) as f:
                return FakeOut(str(sum(1 for _ in f)))
        raise ValueError(cmd)

    monkeypatch.setattr('tgrag.utils.analytics.run', _fake_run)


def test_compute_density_basic():
    assert compute_density(10, 0) == 0.0
    assert compute_density(10, 90) == 1.0
    assert compute_density(5, 5) == 5 / 20


def test_compute_density_edge_cases():
    assert compute_density(0, 10) == 0.0
    assert compute_density(1, 10) == 0.0


def test_count_lines_plain(tmpdir: Path, fake_run):
    p = tmpdir / 'f.txt'
    p.write_text('a\nb\nc\n')
    assert count_lines(str(p)) == 3


def test_count_lines_gzip(tmpdir: Path):
    p = tmpdir / 'f.txt.gz'
    with gzip.open(p, 'wt') as f:
        f.write('x\ny\n')
    assert count_lines(str(p)) == 2


def test_count_sorted_keys_basic(tmpdir: Path):
    inp = tmpdir / 'sorted.txt'
    out = tmpdir / 'counts.tsv'
    inp.write_text('a\na\nb\nb\nb\nc\n')

    n = count_sorted_keys(inp, out)

    assert n == 3
    assert out.read_text().splitlines() == [
        'a\t2',
        'b\t3',
        'c\t1',
    ]


def test_compute_degrees_basic(tmpdir: Path, sample_edges, fake_sort):
    deg_path, E = compute_degrees(sample_edges, tmpdir, sort_cmd='sort', mem='1G')

    assert E == 3
    lines = deg_path.read_text().splitlines()
    assert sorted(lines) == sorted(['a\t2', 'b\t2', 'c\t2'])


def test_stats_prints_summary(
    tmpdir: Path, sample_edges, sample_vertices, fake_sort, fake_run, capsys
):
    deg_path, E = compute_degrees(sample_edges, tmpdir, sort_cmd='sort', mem='1G')
    stats(
        deg_tsv=deg_path,
        E=E,
        vert_path=sample_vertices,
        sort_cmd='sort',
        mem='1G',
        tmpdir=tmpdir,
    )

    out = capsys.readouterr().out.strip()
    assert out.startswith('[STATS]')
    assert 'V=' in out
    assert 'E=3' in out
    assert 'density=' in out


def test_pipeline_end_to_end(
    tmpdir: Path, sample_edges, sample_vertices, fake_sort, fake_run
):
    deg_path, E = compute_degrees(sample_edges, tmpdir, sort_cmd='sort', mem='1G')
    stats(
        deg_tsv=deg_path,
        E=E,
        vert_path=sample_vertices,
        sort_cmd='sort',
        mem='1G',
        tmpdir=tmpdir,
    )
