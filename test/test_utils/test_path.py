import os
from pathlib import Path
import pytest

import tgrag.utils.path as path_mod

@pytest.fixture
def fake_module_path(tmp_path: Path, monkeypatch):
    """
    Force __file__ for the module so get_root_dir/get_curr_parent are testable.
    """
    fake_file = tmp_path / "a" / "b" / "c" / "file.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# dummy", encoding="utf-8")

    monkeypatch.setattr(path_mod, "__file__", str(fake_file))
    return fake_file


def test_get_root_dir(fake_module_path):
    root = path_mod.get_root_dir()
    assert root == fake_module_path.parent.parent.parent


def test_get_curr_parent(fake_module_path):
    parent = path_mod.get_curr_parent()
    assert parent == fake_module_path.parent.parent


def test_get_no_backup():
    assert path_mod.get_no_backup() == Path("/NOBACKUP")


def test_get_cwd(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    assert path_mod.get_cwd() == tmp_path


def test_get_data_root_finds_nearest(tmp_path: Path, monkeypatch):
    base = tmp_path / "proj"
    data_path = base / "data" / "dqr"
    data_path.mkdir(parents=True)
    (data_path / "domain_ratings.csv").write_text("x", encoding="utf-8")

    fake_file = base / "x" / "y" / "file.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# dummy", encoding="utf-8")

    monkeypatch.setattr(path_mod, "__file__", str(fake_file))
    result = path_mod.get_data_root()
    assert result == str(base)


def test_get_data_root_fallback(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(path_mod, "__file__", str(tmp_path / "file.py"))
    monkeypatch.chdir(tmp_path)
    assert path_mod.get_data_root() == str(tmp_path)


def test_get_crawl_data_path_with_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SCRATCH", "/scratch")
    assert path_mod.get_crawl_data_path(tmp_path) == "/scratch/crawl-data"


def test_get_crawl_data_path_without_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SCRATCH", raising=False)
    assert path_mod.get_crawl_data_path(tmp_path) == str(tmp_path / "data" / "crawl-data")


def test_get_wet_file_path_found(tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))

    base = scratch / "crawl-data" / "slice1" / "segments" / "segA" / "wet"
    base.mkdir(parents=True)
    wet = base / "file.warc.wet.gz"
    wet.write_text("x", encoding="utf-8")

    result = path_mod.get_wet_file_path("slice1", "/ignored")
    assert result == str(wet)


def test_get_wet_file_path_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        path_mod.get_wet_file_path("noslice", "/ignored")


def test_get_data_paths():
    v, e = path_mod.get_data_paths("sliceX", "/crawl")
    assert v == "/crawl/sliceX/output/vertices.txt.gz"
    assert e == "/crawl/sliceX/output/edges.txt.gz"