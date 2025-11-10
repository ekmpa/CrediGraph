import sqlite3

import pytest
import torch

from tgrag.dataset.torch_geometric_graph_store import (
    EdgeLayout,
    Rel,
    SQLiteGraphStore,
)


@pytest.fixture(scope='module')
def sqlite_graph_store(tmp_path_factory):
    db_path = tmp_path_factory.mktemp('data') / 'test_graph.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()

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

    for i in range(100_000):
        con.execute(
            'INSERT INTO edges VALUES (?, ?, ?, ?)', (i, i + 1, 'LINKS_TO', 20240505)
        )
    con.commit()
    con.close()
    return SQLiteGraphStore(db_path=db_path)


def test_initialization(sqlite_graph_store) -> None:
    pass


def test_get_size(sqlite_graph_store) -> None:
    size = sqlite_graph_store._get_size(relation='LINKS_TO')
    assert size == (100000, 100000)


def test_populate_edge_attrs(sqlite_graph_store) -> None:
    sqlite_graph_store._populate_edge_attrs()
    s = sqlite_graph_store.store
    for key, value in s.items():
        key == (('domain', 'LINKS_TO', 'domain'), 'coo', True)
        value == Rel(
            ('domain', 'LINKS_TO', 'domain'),
            EdgeLayout.COO,
            True,
            (100000, 100000),
            False,
            None,
        )


def test_get_edge_index(sqlite_graph_store) -> None:
    t = sqlite_graph_store[('domain', 'LINKS_TO', 'domain'), 'coo']
    assert isinstance(t, tuple)
    assert len(t) == 2
    assert all(isinstance(x, torch.Tensor) for x in t)
    assert t[0].size() == torch.Size([100000])
    assert t[1].size() == torch.Size([100000])


def test_get_all_edge_attrs(sqlite_graph_store) -> None:
    edge_attributes = sqlite_graph_store.get_all_edge_attrs()
    assert len(edge_attributes) == 1
    edge_attribute = edge_attributes[0]
    assert isinstance(edge_attribute.edge_type, tuple)
    assert edge_attribute.edge_type == ('domain', 'LINKS_TO', 'domain')

    assert edge_attribute.layout == EdgeLayout.COO

    assert edge_attribute.is_sorted == True

    assert edge_attribute.size == (100000, 100000)


### Tests from PyG 2.0 scalable tutorial ###
## Some tests require the _put_tensor() implementation which is not necessary
## for us yet.


def test_sqlite_graph_store_interface(sqlite_graph_store):
    all_attrs = sqlite_graph_store.get_all_edge_attrs()
    assert len(all_attrs) > 0, 'No edge attributes found — edge table empty?'
    for attr in all_attrs:
        assert isinstance(attr.edge_type, tuple)
        assert attr.layout == EdgeLayout.COO
        assert isinstance(attr.size, tuple)
        assert len(attr.size) == 2

    edge_attr = all_attrs[0]
    edge_index = sqlite_graph_store._get_edge_index(edge_attr)

    assert isinstance(edge_index, tuple)
    assert all(isinstance(t, torch.Tensor) for t in edge_index)
    assert edge_index[0].dtype == torch.int64
    assert edge_index[1].dtype == torch.int64
    assert edge_index[0].shape == edge_index[1].shape
    assert (
        edge_index[0].numel()
        == sqlite_graph_store.cursor.execute(
            'SELECT COUNT(*) FROM edges WHERE relation=?', (edge_attr.edge_type[1],)
        ).fetchone()[0]
    )

    # Test subgraph consitency
    subset = edge_index[0][:5], edge_index[1][:5]
    assert subset[0].shape == subset[1].shape
    assert subset[0].shape[0] == 5

    e = sqlite_graph_store[edge_attr]
    assert torch.equal(e[0], edge_index[0])
    assert torch.equal(e[1], edge_index[1])
