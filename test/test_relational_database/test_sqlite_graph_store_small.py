import sqlite3

import numpy as np
import pytest
import torch

from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore


@pytest.fixture(scope='module')
def sqlite_graph_store(tmp_path_factory):
    """Creates a more realistic and heterogeneous edges table for testing."""
    np.random.seed(42)
    db_path = tmp_path_factory.mktemp('data') / 'test_graph.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Define schema
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            src_id INTEGER,
            dst_id INTEGER,
            relation TEXT,
            ts INTEGER
        )
    """
    )
    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id)')

    con.execute(
        'INSERT INTO edges VALUES (?, ?, ?, ?)',
        (0, 2, 'LINKS_TO', 20240505),
    )

    con.commit()
    return SQLiteGraphStore(db_path=db_path)


def test_get_size(sqlite_graph_store) -> None:
    size = sqlite_graph_store._get_size(relation='LINKS_TO')
    assert size == (1, 1)


def test_get_edge_index(sqlite_graph_store) -> None:
    t = sqlite_graph_store[('domain', 'LINKS_TO', 'domain'), 'coo']
    assert isinstance(t, tuple)
    assert len(t) == 2
    assert all(isinstance(x, torch.Tensor) for x in t)
    assert t[0].size() == t[1].size()
    assert t[0].size() == torch.Size([1])
    assert t[1].size() == torch.Size([1])
