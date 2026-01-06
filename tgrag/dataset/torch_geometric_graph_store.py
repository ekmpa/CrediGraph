import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore
from torch_geometric.typing import EdgeTensorType


class Rel:
    """Container for metadata and cached data of a single edge relation."""

    def __init__(
        self,
        edge_type: Tuple[str, str, str],
        layout: EdgeLayout,
        is_sorted: bool,
        size: Tuple[int, int],
        materialized: bool = False,
        edge_index: Optional[EdgeTensorType] = None,
    ) -> None:
        """Initialize a relation descriptor.

        Parameters:
            edge_type : Tuple[str, str, str]
                Edge type tuple (src_type, relation, dst_type).
            layout : EdgeLayout
                Edge layout format (e.g., COO).
            is_sorted : bool
                Whether the edge index is sorted.
            size : Tuple[int, int]
                Number of unique source and destination nodes.
            materialized : bool, optional
                Whether the edge index has been loaded into memory.
            edge_index : Optional[EdgeTensorType], optional
                Cached edge index tensor.
        """
        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size
        self.materialized = materialized
        self.edge_index = edge_index


class SQLiteGraphStore(GraphStore):
    """GraphStore backed by a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        """Open a SQLite database and initialize edge relation metadata.

        Parameters:
            db_path : Path
                Path to the SQLite database file.
        """
        super().__init__()
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self.cursor = self.con.cursor()

        self.store: Dict[Tuple, Rel] = {}
        self._populate_edge_attrs()

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        """Return a unique key for an edge attribute.

        Parameters:
            attr : EdgeAttr
                Edge attribute descriptor.

        Returns:
            Tuple
                Hashable key identifying the edge attribute.
        """
        return (attr.edge_type, attr.layout.value, attr.is_sorted)

    def _get_size(self, relation: str) -> Tuple[int, int]:
        """Return the number of unique source and destination nodes for a relation.

        Parameters:
            relation : str
                Relation name.

        Returns:
            Tuple[int, int]
                (num_unique_sources, num_unique_destinations)
        """
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
        """Populate the internal relation store from the edges table."""
        self.cursor.execute('SELECT DISTINCT relation FROM edges')
        for row in self.cursor.fetchall():
            rel_name = row['relation']
            edge_type = ('domain', rel_name, 'domain')
            size = self._get_size(rel_name)
            rel = Rel(edge_type, EdgeLayout.COO.value, True, size, False, None)
            key = self.key(EdgeAttr(edge_type, EdgeLayout.COO, True))
            self.store[key] = rel

    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        """Writing edge indices is not supported.

        Raises:
            NotImplementedError
                Always raised when called.
        """
        raise NotImplementedError('Writing edges not supported yet.')

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        """Retrieve or materialize the edge index for a given edge attribute.

        Parameters:
            edge_attr : EdgeAttr
                Edge attribute descriptor.

        Returns:
            Optional[EdgeTensorType]
                Edge index tensors, or None if no edges exist.
        """
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

            rel.edge_index = (src, dst)
            rel.materialized = True

        return rel.edge_index

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        """Remove a cached edge index from the store.

        Parameters:
            edge_attr : EdgeAttr
                Edge attribute descriptor.

        Returns:
            bool
                True if the edge was removed, False otherwise.
        """
        key = self.key(edge_attr)
        if key in self.store:
            del self.store[key]
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """Return all available edge attributes.

        Returns:
            List[EdgeAttr]
                List of edge attribute descriptors.
        """
        return [
            EdgeAttr(rel.edge_type, EdgeLayout.COO, rel.is_sorted, rel.size)
            for rel in self.store.values()
        ]
