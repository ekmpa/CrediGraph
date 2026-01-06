import sqlite3


def table_has_data(con: sqlite3.Connection, table: str) -> bool:
    """Return True if the given table exists and contains at least one row.

    Parameters:
        con : sqlite3.Connection
            Open SQLite connection to query.
        table : str
            Name of the table to check.

    Returns:
        bool
    """
    cur = con.execute(f'SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)')
    return cur.fetchone()[0] == 1


def edge_table_has_data(con: sqlite3.Connection, table: str = 'edges') -> bool:
    """Return True if the edge table exists and contains at least one row.

    Parameters:
        con : sqlite3.Connection
            Open SQLite connection to query.
        table : str, optional
            Name of the edge table to check (default: 'edges').

    Returns:
        bool
    """
    return table_has_data(con, table)


def edge_table_populated(con: sqlite3.Connection) -> bool:
    """Return True if the default edge table ('edges') contains at least one row.

    Parameters:
        con : sqlite3.Connection
            Open SQLite connection to query.

    Returns:
        bool
    """
    return table_has_data(con, 'edges')


def is_db_empty(con: sqlite3.Connection) -> bool:
    """Return True if the database contains no user tables or all user tables are empty.

    Parameters:
        con : sqlite3.Connection
            Open SQLite connection to inspect.

    Returns:
        bool
    """
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [r[0] for r in cur.fetchall()]

    if not tables:
        return True  # no tables means empty DB

    for t in tables:
        if table_has_data(con, t):
            return False

    return True
