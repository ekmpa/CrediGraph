import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore
from torch_geometric.typing import EdgeTensorType


class Rel:
    def __init__(
        self,
        edge_type: Tuple[str, str, str],
        layout: EdgeLayout,
        is_sorted: bool,
        size: Tuple[int, int],
        materialized: bool = False,
        edge_index: Optional[EdgeTensorType] = None,
    ) -> None:
        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size
        self.materialized = materialized
        self.edge_index = edge_index


class SQLiteGraphStore(GraphStore):
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self.cursor = self.con.cursor()

        self.store: Dict[Tuple, Rel] = {}
        self._populate_edge_attrs()

    # Helper: unique key for an edge attribute
    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value, attr.is_sorted)

    def _get_size(self, relation: str) -> Tuple[int, int]:
        """Return the number of unique source and destination nodes for a relation."""
        query = """
            SELECT
                COUNT(DISTINCT src_id),
                COUNT(DISTINCT dst_id)
            FROM edges
            WHERE relation = ?
        """
        self.cursor.execute(query, (relation,))
        row = self.cursor.fetchone()
        src_count, dst_count = row if row else (0, 0)
        return (src_count, dst_count)

    def _populate_edge_attrs(self) -> None:
        self.cursor.execute('SELECT DISTINCT relation FROM edges')
        for row in self.cursor.fetchall():
            rel_name = row['relation']
            edge_type = ('domain', rel_name, 'domain')
            size = self._get_size(rel_name)
            rel = Rel(edge_type, EdgeLayout.COO.value, True, size, False, None)
            key = self.key(EdgeAttr(edge_type, EdgeLayout.COO, True))
            self.store[key] = rel

    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        raise NotImplementedError('Writing edges not supported yet.')

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        if edge_attr.layout.value == EdgeLayout.COO.value:
            if edge_attr.is_sorted == False:
                edge_attr.is_sorted = True

        key = self.key(edge_attr)
        if key not in self.store:
            return None

        rel = self.store[key]
        if rel.layout != EdgeLayout.COO.value:
            raise ValueError('Only COO layout supported in SQLiteGraphStore')

        if not rel.materialized:
            relation = rel.edge_type[1]  # "LINKS_TO"
            self.cursor.execute(
                'SELECT src_id, dst_id FROM edges WHERE relation=?',
                (relation,),
            )
            data = np.array(self.cursor.fetchall(), dtype=np.int64)
            if data.size == 0:
                return None
            # edge_index = torch.from_numpy(data.T)  # shape (2, num_edges)
            src = torch.from_numpy(data[:, 0].copy()).long().contiguous()
            dst = torch.from_numpy(data[:, 1].copy()).long().contiguous()
            edge_index = torch.stack([src, dst], dim=0).contiguous()

            # rel.edge_index = (src, dst)
            rel.edge_index = edge_index
            rel.materialized = True

        return rel.edge_index

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        key = self.key(edge_attr)
        if key in self.store:
            del self.store[key]
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return [
            EdgeAttr(rel.edge_type, EdgeLayout.COO, rel.is_sorted, rel.size)
            for rel in self.store.values()
        ]
