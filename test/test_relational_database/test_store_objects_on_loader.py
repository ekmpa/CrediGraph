import sqlite3

import numpy as np
import pytest
import torch
from torch_geometric.loader import NeighborLoader

from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore
from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore


@pytest.fixture(scope='module')
def sqlite_graph_and_feature_store(tmp_path_factory):
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
    con.execute(
        'INSERT INTO edges VALUES (?, ?, ?, ?)',
        (2, 1, 'LINKS_TO', 20240505),
    )

    record = [
        {'domain': 0, 'ts': 20241111, 'y': -1.0, 'x': [0.1, 0.2, 0.3]},
        {'domain': 1, 'ts': 20241111, 'y': -1.0, 'x': [0.1, -0.04, 0.32]},
        {'domain': 2, 'ts': 20241111, 'y': -1.0, 'x': [0.1, -0.04, 0.32]},
    ]
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

    return SQLiteGraphStore(db_path=db_path), SQLiteFeatureStore(db_path=db_path)


def test_initialization(sqlite_graph_and_feature_store) -> None:
    g, f = sqlite_graph_and_feature_store
    


@pytest.mark.integration
def test_correct_next_iter(sqlite_graph_and_feature_store) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    graph_store, feature_store = sqlite_graph_and_feature_store
    num_neighbors = {key.edge_type: [1] for key in graph_store.get_all_edge_attrs()}
    print(f'num_neighbors: {num_neighbors}')
    loader = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={key.edge_type: [1, 1, 1] for key in graph_store.get_all_edge_attrs()},
        batch_size=1,
        input_nodes=('domain', torch.arange(2)),
        input_time=None,
        num_workers=4,
    )

    for batch in loader:
        print(batch)

