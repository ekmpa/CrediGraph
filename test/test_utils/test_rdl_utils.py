import sqlite3

import numpy as np
import pytest

from tgrag.utils.rd_utils import is_db_empty, table_has_data


@pytest.fixture(scope='module')
def initialize_db_return_con(tmp_path_factory):
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
    return con


@pytest.fixture(scope='module')
def initialize_db_return_con_empty(tmp_path_factory):
    """Creates a more realistic and heterogeneous edges table for testing."""
    np.random.seed(42)
    db_path = tmp_path_factory.mktemp('data') / 'test_graph_empty.db'
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

    con.commit()
    return con


@pytest.fixture(scope='module')
def initialize_db_no_edges_con(tmp_path_factory):
    """Creates a more realistic and heterogeneous edges table for testing."""
    np.random.seed(42)
    db_path = tmp_path_factory.mktemp('data') / 'test_graph_no_edges.db'
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
        x = np.random.randn(128).astype(np.float32).tobytes()
        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)', (src, src, x, float(src % 5))
        )
        con.execute(
            'INSERT INTO domain VALUES (?, ?, ?, ?)',
            (int(dst), int(dst), x, float(dst % 5)),
        )

    con.commit()
    return con


def test_is_db_empty(initialize_db_return_con):
    con = initialize_db_return_con
    assert is_db_empty(con) == False


def test_has_data(initialize_db_return_con):
    con = initialize_db_return_con
    assert table_has_data(con=con, table='domain') == True
    assert table_has_data(con=con, table='edges') == True


def test_has_data_empty(initialize_db_return_con_empty):
    con = initialize_db_return_con_empty
    assert table_has_data(con=con, table='domain') == False
    assert table_has_data(con=con, table='edges') == False


def test_has_data_no_edges(initialize_db_no_edges_con):
    con = initialize_db_no_edges_con
    assert table_has_data(con=con, table='domain') == True
    assert table_has_data(con=con, table='edges') == False


def test_is_db_empty_on_empty(initialize_db_return_con_empty):
    con = initialize_db_return_con_empty
    assert is_db_empty(con) == True
