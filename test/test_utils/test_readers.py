import csv
import gzip
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch

from tgrag.utils.readers import (
    line_reader,
    read_vertex_file,
    read_edge_file,
    load_edges,
    load_node_domain_map,
    load_node_csv,
    load_edge_csv,
    load_large_edge_csv,
    load_target_nids,
    load_domains,
    read_weak_labels,
    read_reg_scores,
    collect_merged,
    get_seed_embeddings,
)

@pytest.fixture
def tmpdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def simple_txt(tmpdir: Path) -> Path:
    p = tmpdir / "f.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    return p


@pytest.fixture
def simple_gz(tmpdir: Path) -> Path:
    p = tmpdir / "f.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("x\ny\n")
    return p


@pytest.fixture
def vertices_file(tmpdir: Path) -> Path:
    p = tmpdir / "verts.txt"
    p.write_text("a\nb\nb\nc\n\n", encoding="utf-8")
    return p


@pytest.fixture
def edges_file(tmpdir: Path) -> Path:
    p = tmpdir / "edges.txt"
    p.write_text("0\t1\n1\t2\n2\t0\n", encoding="utf-8")
    return p


@pytest.fixture
def id_map() -> Dict[int, str]:
    return {0: "a.com", 1: "b.com", 2: "c.com"}

def test_line_reader_plain(simple_txt: Path):
    assert list(line_reader(simple_txt)) == ["a", "b", "c"]


def test_line_reader_gzip(simple_gz: Path):
    assert list(line_reader(simple_gz)) == ["x", "y"]


def test_read_vertex_file(vertices_file: Path):
    assert read_vertex_file(str(vertices_file)) == {"a", "b", "c"}


def test_read_edge_file(edges_file: Path, id_map):
    edges = read_edge_file(str(edges_file), id_map)
    assert edges == {("a.com", "b.com"), ("b.com", "c.com"), ("c.com", "a.com")}


def test_load_edges(tmpdir: Path):
    p = tmpdir / "edges.txt"
    p.write_text("a b\nb c\nbad bad line\n", encoding="utf-8")
    assert load_edges(str(p)) == [("a", "b"), ("b", "c")]


def test_load_node_domain_map(tmpdir: Path):
    p = tmpdir / "nodes.txt"
    p.write_text("0 a.com\n1 b.com\n", encoding="utf-8")

    id_to_dom, dom_to_id = load_node_domain_map(p)
    assert id_to_dom == {"0": "a.com", "1": "b.com"}
    assert dom_to_id == {"a.com": "0", "b.com": "1"}


def test_load_node_csv_basic(tmpdir: Path):
    p = tmpdir / "nodes.csv"
    p.write_text("nid,feat\n0,1\n1,2\n", encoding="utf-8")

    x, mapping, idx = load_node_csv(str(p), index_col=0, encoders=None)
    assert x is None
    assert mapping == {0: 0, 1: 1}
    assert len(idx) == 2


def test_load_edge_csv_basic(tmpdir: Path):
    p = tmpdir / "edges.csv"
    p.write_text("src,dst\n0,1\n1,0\n", encoding="utf-8")

    mapping = {0: 0, 1: 1}
    edge_index, edge_attr = load_edge_csv(str(p), "src", "dst", mapping)

    assert edge_attr is None
    assert edge_index.tolist() == [[0, 1], [1, 0]]


def test_load_large_edge_csv_basic(tmpdir: Path):
    p = tmpdir / "edges.csv"
    p.write_text("src,dst\n0,1\n1,0\n", encoding="utf-8")

    mapping = {0: 0, 1: 1}
    edge_index, edge_attr = load_large_edge_csv(str(p), "src", "dst", mapping)

    assert edge_attr is None
    assert edge_index.tolist() == [[0, 1], [1, 0]]


def test_load_target_nids(tmpdir: Path):
    p = tmpdir / "targets.csv"
    p.write_text("nid,val\n1,x\n2,y\n", encoding="utf-8")

    assert load_target_nids(str(p)) == {1, 2}


def test_load_domains(tmpdir: Path):
    p = tmpdir / "d.csv"
    p.write_text("domain\nWWW.A.COM\nb.com\n\n", encoding="utf-8")

    assert load_domains(p) == {"a.com", "b.com"}


def test_read_weak_labels(tmpdir: Path):
    p = tmpdir / "w.csv"
    p.write_text("domain,label\na.com,1\nb.com,0\n", encoding="utf-8")

    assert read_weak_labels(p) == {"a.com": 1, "b.com": 0}


def test_read_reg_scores(tmpdir: Path):
    p = tmpdir / "r.csv"
    p.write_text("domain,pc1\na.com,0.1\nb.com,x\n", encoding="utf-8")

    assert read_reg_scores(p) == {"a.com": 0.1}


def test_collect_merged(tmpdir: Path):
    p1 = tmpdir / "a.csv"
    p2 = tmpdir / "b.csv"
    out = tmpdir / "out.csv"

    p1.write_text("domain,label\na.com,1\nb.com,0\n", encoding="utf-8")
    p2.write_text("domain,label\na.com,0\nc.com,1\n", encoding="utf-8")

    result = collect_merged([p1, p2, out], out)

    assert result == {
        "a.com": [1.0, 0.0],
        "b.com": [0.0],
        "c.com": [1.0],
    }


def test_get_seed_embeddings(monkeypatch, tmpdir: Path):
    fake_root = tmpdir
    data_path = fake_root / "data.pkl"

    with open(data_path, "wb") as f:
        pickle.dump({"a.com": [1.0, 2.0]}, f)

    monkeypatch.setattr(
        "tgrag.utils.readers.get_root_dir",
        lambda: fake_root,
    )

    emb = get_seed_embeddings(file_name="data.pkl")
    assert isinstance(emb["a.com"], torch.Tensor)
    assert emb["a.com"].tolist() == [1.0, 2.0]