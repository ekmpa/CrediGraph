import sqlite3


def table_has_data(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute(f'SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)')
    return cur.fetchone()[0] == 1


def edge_table_has_data(con: sqlite3.Connection, table: str = 'edges') -> bool:
    return table_has_data(con, table)


def edge_table_populated(con: sqlite3.Connection) -> bool:
    return table_has_data(con, 'edges')


def is_db_empty(con: sqlite3.Connection) -> bool:
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
