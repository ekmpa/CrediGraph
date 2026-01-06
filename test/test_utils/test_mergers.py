import gzip
from pathlib import Path
import pytest

from tgrag.utils.mergers import (
    append_target_nodes,
    append_target_edges,
    extract_all_domains,
    merge_processed_labels,
    merge_reg_class,
)


@pytest.fixture
def tmpdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def fake_executor(monkeypatch):
    """Force ProcessPoolExecutor to run synchronously for tests."""
    class FakeFuture:
        def __init__(self, result):
            self._result = result
        def result(self):
            return self._result

    class FakeExecutor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args):
            return FakeFuture(fn(*args))

    monkeypatch.setattr(
        "tgrag.utils.mergers.ProcessPoolExecutor", FakeExecutor
    )
    monkeypatch.setattr(
        "tgrag.utils.mergers.as_completed", lambda fs: fs
    )


def write_gz(path: Path, text: str):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


def read_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def test_append_target_nodes_basic(tmpdir: Path, fake_executor):
    src = tmpdir / "src"
    tgt = tmpdir / "tgt"
    (src / "vertices").mkdir(parents=True)
    tgt.mkdir()

    write_gz(src / "vertices" / "v1.txt.gz", "1\ta.com\n2\tb.com\n")
    write_gz(src / "vertices" / "v2.txt.gz", "3\tc.com\n")

    id_map = append_target_nodes(str(src), str(tgt))

    assert id_map == {1: "a.com", 2: "b.com", 3: "c.com"}

    verts = read_gz(tgt / "vertices.txt.gz")
    assert set(verts) == {"a.com", "b.com", "c.com"}


def test_append_target_nodes_no_files(tmpdir: Path, capsys):
    src = tmpdir / "src"
    tgt = tmpdir / "tgt"
    (src / "vertices").mkdir(parents=True)
    tgt.mkdir()

    result = append_target_nodes(str(src), str(tgt))

    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert result == {}


def test_append_target_edges_basic(tmpdir: Path, fake_executor):
    src = tmpdir / "src"
    tgt = tmpdir / "tgt"
    (src / "edges").mkdir(parents=True)
    tgt.mkdir()

    id_to_domain = {1: "a.com", 2: "b.com", 3: "c.com"}

    write_gz(src / "edges" / "e1.txt.gz", "1\t2\n2\t3\n")
    write_gz(src / "edges" / "e2.txt.gz", "1\t3\n")

    append_target_edges(str(src), str(tgt), id_to_domain)

    edges = read_gz(tgt / "edges.txt.gz")
    assert set(edges) == {"a.com\tb.com", "b.com\tc.com", "a.com\tc.com"}


def test_append_target_edges_no_files(tmpdir: Path, capsys):
    src = tmpdir / "src"
    tgt = tmpdir / "tgt"
    (src / "edges").mkdir(parents=True)
    tgt.mkdir()

    append_target_edges(str(src), str(tgt), {})

    out = capsys.readouterr().out
    assert "[WARN]" in out


def test_extract_all_domains(tmpdir: Path):
    verts = tmpdir / "verts.txt"
    edges = tmpdir / "edges.txt"
    out = tmpdir / "domains.txt"

    verts.write_text("a.com\nb.com\n\n", encoding="utf-8")
    edges.write_text("a.com\tb.com\nb.com\tc.com\nbad\n", encoding="utf-8")

    extract_all_domains(str(verts), str(edges), str(out))

    assert set(out.read_text().splitlines()) == {"a.com", "b.com", "c.com"}


def test_merge_processed_labels(tmpdir: Path, monkeypatch):
    csv1 = tmpdir / "a.csv"
    csv2 = tmpdir / "b.csv"
    out = tmpdir / "out.csv"

    csv1.write_text("domain,label\na.com,1\nb.com,0\n", encoding="utf-8")
    csv2.write_text("domain,label\na.com,0\nc.com,1\n", encoding="utf-8")

    called = {}

    def fake_write(domain_labels, output_csv):
        called["data"] = domain_labels
        called["out"] = output_csv

    monkeypatch.setattr("tgrag.utils.mergers.write_aggr_labelled", fake_write)
    monkeypatch.setattr("tgrag.utils.mergers.check_processed_labels", lambda _: None)

    merge_processed_labels(tmpdir, out)

    assert called["out"] == out
    assert called["data"] == {
        "a.com": [1.0, 0.0],
        "b.com": [0.0],
        "c.com": [1.0],
    }


def test_merge_reg_class(tmpdir: Path):
    weak = tmpdir / "weak.csv"
    reg = tmpdir / "reg.csv"
    out = tmpdir / "merged.csv"

    weak.write_text("domain,label\na.com,1\nb.com,0\n", encoding="utf-8")
    reg.write_text("domain,pc1\na.com,0.5\nc.com,0.2\n", encoding="utf-8")

    merge_reg_class(weak, reg, out)

    rows = out.read_text().splitlines()
    assert rows == [
        "domain,weak_label,reg_score",
        "a.com,1,0.5",
        "b.com,0,",
        "c.com,,0.2",
    ]