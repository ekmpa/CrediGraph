import sqlite3
import time

import numpy as np
import pytest
import torch
from torch_geometric.data import TensorAttr

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore


@pytest.fixture(scope="module")
def sqlite_feature_store(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("data") / "test_graph.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE domain (id INTEGER PRIMARY KEY,ts INTEGER, x BLOB, y REAL)"
    )
    for i in range(100_000):
        x = np.random.randn(128).astype(np.float32).tobytes()
        con.execute("INSERT INTO domain VALUES (?, ?, ?, ?)", (i, i, x, float(i % 5)))
    con.commit()
    con.close()
    return SQLiteFeatureStore(db_path=db_path, read_only=True)


def test_random_access_efficiency(sqlite_feature_store):
    random_ids = torch.randperm(100_000)[:1024]

    start = time.perf_counter()
    tensor = sqlite_feature_store["domain", "x", random_ids]
    elapsed = time.perf_counter() - start

    assert tensor.shape == (1024, 128)
    assert elapsed < 0.05, f"Random access too slow: {elapsed:.3f}s"
    print(elapsed)
