import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore


@pytest.fixture
def db_path(tmp_path: Path):
    db_file = tmp_path / 'test_graph.db'

    record = [
        {'domain': 0, 'ts': 20241111, 'y': -1.0, 'x': [0.1, 0.2, 0.3]},
        {'domain': 1, 'ts': 20241111, 'y': -1.0, 'x': [0.1, -0.04, 0.32]},
    ]
    con = sqlite3.connect(db_file)
    con.execute(
        'CREATE TABLE domain(id INTEGER PRIMARY KEY, ts INTEGER, x BLOB, y REAL)'
    )
    for r in record:
        x = np.array(r['x'], dtype=np.float32).tobytes()

        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)',
            (int(r['domain']), int(r['ts']), x, float(r['y'])),
        )
    con.commit()
    con.close()

    return db_file


def test_get_tensor_single_index(db_path) -> None:
    feature_store = SQLiteFeatureStore(db_path=db_path)
    x_features_0 = feature_store['domain', 'x', 0]
    assert torch.equal(x_features_0, torch.tensor([[0.1, 0.2, 0.3]]))

    x_features_1 = feature_store['domain', 'x', 1]
    assert torch.equal(x_features_1, torch.tensor([[0.1, -0.04, 0.32]]))
