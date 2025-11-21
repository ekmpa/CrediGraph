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
    """Creates a more realistic and heterogeneous edges table for testing."""
    np.random.seed(42)
    db_path = tmp_path_factory.mktemp('data') / 'test_graph.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Define schema
    con.execute(
        'CREATE TABLE domain(id INTEGER PRIMARY KEY, ts INTEGER, x BLOB, y REAL)'
    )
    cur.execute(
        """
        CREATE TABLE edges (
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

    # --- RELATION 1: LINKS_TO ---
    # Non-contiguous src/dst IDs and sparse nodes
    for i in range(0, 50_000, 2):
        src = i * 2 + 10  # start at 10, skip every other ID
        dst = src + np.random.choice([3])
        if 0.5 <= np.random.rand():
            con.execute(
                'INSERT INTO edges VALUES (?, ?, ?, ?)',
                (src, int(dst), 'LINKS_TO', 20240505),
            )
        x = np.random.randn(128).astype(np.float32).tobytes()
        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)', (src, src, x, float(src % 5))
        )
        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)',
            (int(dst), int(dst), x, float(dst % 5)),
        )

    con.commit()
    return SQLiteGraphStore(db_path=db_path), SQLiteFeatureStore(db_path=db_path)


def test_initialization(sqlite_graph_and_feature_store) -> None:
    g, f = sqlite_graph_and_feature_store


def test_sampler(sqlite_graph_and_feature_store) -> None:
    graph_store, feature_store = sqlite_graph_and_feature_store
    sampler = SQLiteNeighborSampler(
        graph_store, num_neighbors={('domain', 'LINKS_TO', 'domain'): [5]}
    )

    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=sampler,
        batch_size=3,
        input_nodes=('domain', torch.tensor([87530])),
    )

    print(next(iter(loader)))

    for batch in loader:
        print(
            f'Edge index -- Batch: {batch["domain", "LINKS_TO", "domain"].edge_index}'
        )
        print(f'Features -- Batch: {batch["domain"]["x"]}')
        print(f'Features -- Batch: {batch["domain"]["y"]}')


def test_sampler_no_edge_exists(sqlite_graph_and_feature_store) -> None:
    graph_store, feature_store = sqlite_graph_and_feature_store
    sampler = SQLiteNeighborSampler(
        graph_store, num_neighbors={('domain', 'LINKS_TO', 'domain'): [5]}
    )

    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=sampler,
        batch_size=3,
        input_nodes=('domain', torch.tensor([10])),
    )

    print(next(iter(loader)))

    for batch in loader:
        print(
            f'Edge index -- Batch: {batch["domain", "LINKS_TO", "domain"].edge_index}'
        )
        print(f'Features -- Batch: {batch["domain"]["x"]}')
        print(f'Features -- Batch: {batch["domain"]["y"]}')
