import multiprocessing

from torch_geometric.data.graph_store import GraphStore


class CustomGraphStore(GraphStore):
    def __init__(self, num_threads: int) -> None:
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.num_threads = num_threads
