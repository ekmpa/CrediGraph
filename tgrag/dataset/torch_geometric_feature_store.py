import pickle
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType


class SQLiteFeatureStore(FeatureStore):
    def __init__(self, db_path: Path, read_only: bool = True):
        super().__init__()
        self.db_path = db_path
        uri = f'file:{db_path}?mode={"ro" if read_only else "rwc"}'
        self.con = sqlite3.connect(uri, uri=True)
        self.con.row_factory = sqlite3.Row
        self.cursor = self.con.cursor()

    def _get_table_columns(self, table: str) -> list[str]:
        q = f'PRAGMA table_info({table})'
        self.cursor.execute(q)
        return [r['name'] for r in self.cursor.fetchall()]

    def _deserialize(self, blob: Any) -> None | int | float | np.ndarray:
        if blob is None:
            return None
        if isinstance(blob, (int, float)):
            return blob
        try:
            return pickle.load(blob)
        except Exception:
            return np.frombuffer(blob, dtype=np.float32)

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        raise NotImplementedError('Writing not yet supported.')

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        table_name = attr.group_name
        attr_name = attr.attr_name
        idx = attr.index

        q = f'SELECT {attr_name} FROM {table_name}'
        params = []
        if idx is not None:
            if isinstance(idx, int):
                q += ' WHERE id = ?'
                params = [idx]

            elif isinstance(idx, slice):
                q += ' WHERE id BETWEEN ? and ?'
                params = [idx.start, idx.stop - 1]

            elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
                ids = [int(i) for i in idx]
                placeholders = ','.join('?' * len(ids))
                q += f' WHERE id IN ({placeholders})'
                params = ids

        self.cursor.execute(q, params)
        rows = [r[0] for r in self.cursor.fetchall()]
        tensors = [self._deserialize(r) for r in rows]
        if not tensors:
            return None
        if isinstance(tensors[0], np.ndarray):
            try:
                return torch.from_numpy(np.stack(tensors))
            except ValueError:
                return torch.from_numpy(np.array(tensors, dtype=np.float32))
        elif isinstance(tensors[0], (list, tuple)):
            return torch.tensor(np.array(tensors, dtype=np.float32))
        else:
            return torch.tensor(tensors)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise NotImplementedError('Removal not yet supported.')

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        table_name = attr.group_name
        attr_name = attr.attr_name

        q = f'SELECT COUNT (*) FROM {table_name}'
        self.cursor.execute(q)
        n = self.cursor.fetchone()[0]

        q = f'SELECT {attr_name} FROM {table_name} WHERE {attr_name} IS NOT NULL LIMIT 1'
        self.cursor.execute(q)
        row = self.cursor.fetchone()

        if row is None or row[0] is None:
            return (n,)

        if isinstance(row[0], (int, float, str)):
            return (n, 1)

        sample = self._deserialize(row[0])

        if isinstance(sample, np.ndarray):
            return (n,) + sample.shape
        elif isinstance(sample, (list, tuple)):
            return (n, len(sample))
        elif np.isscalar(sample):
            return (n, 1)
        else:
            return (n,)

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        tables = self._get_all_tables()
        all_attrs = []
        for t in tables:
            for col in self._get_table_columns(t):
                all_attrs.append(TensorAttr(group_name=t, attr_name=col))

        return all_attrs

    def _get_all_tables(self) -> List[str]:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in self.cursor.fetchall()]
