import sqlite3
from pathlib import Path

import numpy as np
import pytest
from torch_geometric.data.feature_store import TensorAttr

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore


@pytest.fixture
def db_path(tmp_path: Path):
    db_file = tmp_path / 'test_graph.db'

    record = [
        {'domain': '0.0', 'ts': 20241111, 'y': -1.0, 'x': [0.1, 0.2, 0.3]},
        {'domain': 'google.com', 'ts': 20241111, 'y': -1.0, 'x': [0.1, -0.04, 0.32]},
    ]
    con = sqlite3.connect(db_file)
    con.execute(
        'CREATE TABLE domain(name TEXT PRIMARY KEY, ts INTEGER, x BLOB, y REAL)'
    )
    for r in record:
        x = np.array(r['x'], dtype=np.float32).tobytes()

        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)',
            (str(r['domain']), int(r['ts']), x, float(r['y'])),
        )
    con.commit()
    con.close()

    return db_file


def test_get_all_tensor_attrs(db_path) -> None:
    feature_store = SQLiteFeatureStore(db_path=db_path)

    attrs = feature_store.get_all_tensor_attrs()
    assert len(attrs) == 4
    print(attrs)


def test_get_tensor_size(db_path) -> None:
    t = TensorAttr(group_name='domain', attr_name='x')
    feature_store = SQLiteFeatureStore(db_path=db_path)

    n = feature_store._get_tensor_size(attr=t)
    assert n == (2, 3)

    t = TensorAttr(group_name='domain', attr_name='ts')
    n = feature_store._get_tensor_size(attr=t)
    assert n == (2, 1)

    t = TensorAttr(group_name='domain', attr_name='y')
    n = feature_store._get_tensor_size(attr=t)
    assert n == (2, 1)

    t = TensorAttr(group_name='domain', attr_name='name')
    n = feature_store._get_tensor_size(attr=t)
    assert n == (2, 1)


def test_get_table_columns(db_path) -> None:
    feature_store = SQLiteFeatureStore(db_path=db_path)
    columns = feature_store._get_table_columns(table='domain')
    assert len(columns) == 4
    assert columns[0] == 'name'
    assert columns[1] == 'ts'
    assert columns[2] == 'x'
    assert columns[3] == 'y'


def test_get_tensor(db_path) -> None:
    feature_store = SQLiteFeatureStore(db_path=db_path)

    feature_store['domain', 'x', 0]
