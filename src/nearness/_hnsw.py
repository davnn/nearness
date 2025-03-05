from hnswlib import BFIndex, Index
from safecheck import Float, Float32, NumpyArray, UInt64, typecheck
from typing_extensions import Literal

from ._base import NearestNeighbors


class HNSWNeighbors(NearestNeighbors):
    """Approximate nearest neighbors based on HNSWlib."""

    available_metrics = Literal["l2", "ip", "cosine"]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "l2",
        n_index_neighbors: int = 256,
        n_search_neighbors: int | None = None,
        n_links: int = 16,
        n_threads: int = -1,
        random_seed: int = 0,
        use_bruteforce: bool = False,
    ) -> None:
        """Instantiate HNSW nearest neighbors.

        :param metric: One of ["l2", "ip", "cosine"].
        :param n_index_neighbors: Size of the dynamic neighbors candidate list during index construction.
        :param n_search_neighbors: Size of the dynamic neighbors candidate list during index search.
        :param n_links: Number of connections per node in the graph. Higher values improve accuracy but use more memory.
        :param n_threads: Number of threads to use during index search.
        :param random_seed: Seed for random number generation, ensuring reproducibility across runs.
        :param use_bruteforce: Skip index creation and use bruteforce search over all items instead.
        """
        super().__init__()
        self._index_constructor = BFIndex if use_bruteforce else Index
        self._model = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "HNSWNeighbors":
        n_samples, n_dim = data.shape
        index = self._index_constructor(space=self.parameters.metric, dim=n_dim)

        if self.parameters.use_bruteforce:
            index.init_index(max_elements=n_samples)
            index.add_items(data)
        else:
            index.init_index(
                max_elements=n_samples,
                ef_construction=self.parameters.n_index_neighbors,
                M=self.parameters.n_links,
                random_seed=self.parameters.random_seed,
            )
            index.add_items(data)

        if (n_search := self.parameters.n_search_neighbors) is not None:
            index.set_ef(n_search)

        self._model = index
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[UInt64[NumpyArray, "{n_neighbors}"], Float32[NumpyArray, "{n_neighbors}"]]:
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors)
        return idx.ravel(), dist.ravel()

    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[UInt64[NumpyArray, "m {n_neighbors}"], Float32[NumpyArray, "m {n_neighbors}"]]:
        idx, dist = self._model.knn_query(points, k=n_neighbors, num_threads=self.parameters.n_threads)
        return idx, dist
