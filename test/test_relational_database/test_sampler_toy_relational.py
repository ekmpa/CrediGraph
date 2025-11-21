import sqlite3

import numpy as np
import pytest
import torch
from torch_geometric.loader import NodeLoader

from tgrag.dataset.sampler import SQLiteNeighborSampler
from tgrag.dataset.torch_geometric_feature_store import SQLiteFeatureStore
from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore


@pytest.fixture(scope='module')
def sqlite_graph_and_feature_store(tmp_path_factory):
    """Creates a realistic SQLite graph + feature store with proper remapping:

    domain table:
      row_id  INTEGER PRIMARY KEY AUTOINCREMENT   ← internal row index
      id      INTEGER                             ← external domain id
      ts      INTEGER
      x       BLOB
      y       REAL

    Edges reference domain.id (external ids), not row ids.
    FeatureStore internally indexes on row_id.
    """
    np.random.seed(42)

    db_dir = tmp_path_factory.mktemp('data')
    db_path = db_dir / 'test_graph.db'

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # -------------------------
    # Create schema with explicit AUTOINCREMENT row_id
    # -------------------------
    con.execute(
        """
        CREATE TABLE domain(
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            id INTEGER,
            ts INTEGER,
            x BLOB,
            y REAL
        )
        """
    )

    con.execute(
        """
        CREATE TABLE edges(
            src_id INTEGER,
            dst_id INTEGER,
            relation TEXT,
            ts INTEGER
        )
        """
    )

    con.execute('CREATE INDEX idx_edges_relation ON edges(relation)')
    con.execute('CREATE INDEX idx_edges_src ON edges(src_id)')
    con.execute('CREATE INDEX idx_edges_dst ON edges(dst_id)')

    # -------------------------
    # Insert nodes + edges using EXTERNAL ids (10, 14, 18, ...)
    # Internally row_id will be 1,2,3,... and mapped back.
    # -------------------------
    for i in range(0, 40, 2):
        dom_id = i * 2 + 10  # external domain ID
        dst_id = dom_id + 3

        # Maybe insert an edge:
        con.execute(
            'INSERT INTO edges VALUES (?, ?, ?, ?)',
            (dom_id, dst_id, 'LINKS_TO', 20240505),
        )

        # Insert src and dst domain rows (row_id autoincrement)
        x_blob = np.random.randn(128).astype(np.float32).tobytes()

        con.execute(
            'INSERT INTO domain(id, ts, x, y) VALUES (?, ?, ?, ?)',
            (dom_id, dom_id, x_blob, float(dom_id % 5)),
        )
        con.execute(
            'INSERT INTO domain(id, ts, x, y) VALUES (?, ?, ?, ?)',
            (dst_id, dst_id, x_blob, float(dst_id % 5)),
        )

    con.commit()

    # ---------------
    # Debug: verify mapping
    # ---------------
    rows = cur.execute(
        'SELECT row_id, id FROM domain ORDER BY row_id LIMIT 10'
    ).fetchall()
    print('RowID → DomainID mapping sample:', rows)

    # -------------------------
    # Now initialize stores
    # -------------------------
    graph_store = SQLiteGraphStore(db_path=db_path)
    feature_store = SQLiteFeatureStore(db_path=db_path)

    return graph_store, feature_store


def test_initialization(sqlite_graph_and_feature_store):
    graph_store, feature_store = sqlite_graph_and_feature_store
    assert graph_store is not None
    assert feature_store is not None


def test_sampler(sqlite_graph_and_feature_store):
    graph_store, feature_store = sqlite_graph_and_feature_store

    sampler = SQLiteNeighborSampler(
        graph_store,
        num_neighbors={('domain', 'LINKS_TO', 'domain'): [5]},
    )

    # External domain IDs
    input_ids = torch.tensor([10, 14, 18], dtype=torch.long)

    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=sampler,
        batch_size=3,
        input_nodes=('domain', input_ids),
    )

    batch = next(iter(loader))

    print('\n=== Batch ===')
    print(batch)

    print('\nEdge index:\n', batch['domain', 'LINKS_TO', 'domain'].edge_index)
    print('\nFeatures shape:\n', batch['domain']['x'].shape)

    assert batch['domain']['x'].shape[0] >= 3
