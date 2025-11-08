import sqlite3

import numpy as np


def test_numpy_type_insert():
    record = {'domain': '0.0', 'ts': 20241111, 'y': -1.0, 'x': [0.1, 0.2, 0.3]}
    x = np.array(record['x'], dtype=np.float32).tobytes()

    con = sqlite3.connect(':memory:')
    con.execute(
        'CREATE TABLE domain(name TEXT PRIMARY KEY, ts INTEGER, x BLOB, y REAL)'
    )
    con.execute(
        'INSERT INTO domain VALUES (?, ?, ?, ?)',
        (str(record['domain']), int(record['ts']), x, float(record['y'])),
    )


def test_numpy_type_insert_float():
    record = {
        'domain': 'www.google.com',
        'ts': 20241111,
        'y': 0.5,
        'x': [0.1, 0.2, 0.3],
    }
    x = np.array(record['x'], dtype=np.float32).tobytes()

    con = sqlite3.connect(':memory:')
    con.execute(
        'CREATE TABLE domain(name TEXT PRIMARY KEY, ts INTEGER, x BLOB, y REAL)'
    )
    con.execute(
        'INSERT INTO domain VALUES (?, ?, ?, ?)',
        (str(record['domain']), int(record['ts']), x, float(record['y'])),
    )
