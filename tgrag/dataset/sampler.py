from typing import Dict, List, Tuple, Union

import torch
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.base import EdgeSamplerInput

from tgrag.dataset.torch_geometric_graph_store import SQLiteGraphStore


class SQLiteNeighborSampler(BaseSampler):
    """Neighbor sampler that draws edges from a SQLite-backed graph store."""

    def __init__(
        self,
        graph_store: SQLiteGraphStore,
        num_neighbors: Dict[Tuple[str, str, str], List[int]],
    ):
        """Initialize the sampler with a graph store and neighbor limits.

        Parameters:
            graph_store : SQLiteGraphStore
                SQLite-backed graph storage providing an open cursor.
            num_neighbors : Dict[Tuple[str, str, str], List[int]]
                Mapping from edge type to a list of neighbor counts per hop.
        """
        super().__init__()
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors

    def sample_from_nodes(
        self, index: NodeSamplerInput, **kwargs: str
    ) -> SamplerOutput:
        """Sample neighbors for a batch of seed nodes.

        Parameters:
            index : NodeSamplerInput
                Sampler input containing seed node IDs.
            **kwargs : str
                Unused additional arguments.

        Returns:
            SamplerOutput
                HeteroSamplerOutput containing sampled nodes, edges, and batch mapping.
        """
        seed_nodes = index.node
        seed_type = ('domain', 'LINKS_TO', 'domain')
        k = self.num_neighbors[seed_type][0]

        src_nodes, dst_nodes = [], []

        for node_id in seed_nodes.tolist():
            query = """
                SELECT dst_id
                FROM edges
                WHERE src_id = ?
                ORDER BY RANDOM()
                LIMIT ?
            """
            rows = self.graph_store.cursor.execute(query, (node_id, k)).fetchall()
            for (dst,) in rows:
                src_nodes.append(node_id)
                dst_nodes.append(dst)

        src = torch.tensor(src_nodes, dtype=torch.long)
        dst = torch.tensor(dst_nodes, dtype=torch.long)

        n_id = torch.unique(
            torch.cat([seed_nodes, src, dst])
        )  # This is needed in the case src, dst == []

        node_map = {nid: i for i, nid in enumerate(n_id.tolist())}
        row = torch.tensor([node_map[s.item()] for s in src], dtype=torch.long)
        col = torch.tensor([node_map[d.item()] for d in dst], dtype=torch.long)

        e_id = torch.arange(row.numel(), dtype=torch.long)

        batch = torch.arange(seed_nodes.size(0), dtype=torch.long)

        edge_type = ('domain', 'LINKS_TO', 'domain')

        return HeteroSamplerOutput(
            node={'domain': n_id},  # Global node IDs in this sample
            row={edge_type: row},  # Re-indexed source indices
            col={edge_type: col},  # Re-indexed destination indices
            edge={edge_type: e_id},  # Optional edge IDs (here sequential)
            batch={'domain': batch},  # Seed-to-batch mapping
            metadata=(seed_nodes, None),
        )

    def sample_from_edges(
        self, index: EdgeSamplerInput, **kwargs: str
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        """Edge-based sampling is not supported for this sampler.

        Raises:
            NotImplementedError
                Always raised when this method is called.
        """
        raise NotImplementedError('Sample from edges not implemented')
