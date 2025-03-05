import faiss
from safecheck import Float, Float32, Int64, NumpyArray, typecheck
from typing_extensions import Any, TypeVar

from ._base import NearestNeighbors
from ._base._helpers import IndexWrapper

__all__ = ["FaissIndex", "FaissNeighbors"]

T = TypeVar("T", bound=type[faiss.Index])


class FaissIndex(IndexWrapper[T]):
    def __init__(self, index: T, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.index = index


class FaissNeighbors(NearestNeighbors):
    """A simple wrapper around ``faiss.Index``.

    A Faiss index is built from any combination of the following.

    1. Vector transform — a pre-processing step applied to vectors before indexing (PCA, OPQ).
    2. Coarse quantizer — rough organization of vectors to subdomains (for restricting search
       scope, includes IVF, IMI, and HNSW).
    3. Fine quantizer — a finer compression of vectors into smaller domains (for compressing
       index size, such as PQ).
    4. Refinement — a final step at search-time which re-orders results using distance calculations
       on the original flat vectors. Alternatively, another index (non-flat) index can be used.

    Note that coarse quantization refers to the 'clustering' of vectors (such as inverted indexing with IVF).
    By using coarse quantization, we enable non-exhaustive search by limiting the search scope.
    Fine quantization describes the compression of vectors into codes (as with PQ). The purpose fine quantization
    is to reduce the memory usage of the index.

    References
    ----------
        - https://github.com/facebookresearch/faiss
        - https://www.pinecone.io/learn/series/faiss/

    """

    @typecheck
    def __init__(
        self,
        *,
        index: str | FaissIndex | faiss.Index = "Flat",
        add_data_on_fit: bool = True,
    ) -> None:
        """Instantiate faiss nearest neighbors.

        :param index: An index factory string, a FaissIndex wrapped index or a faiss.Index.
        :param add_data_on_fit: Add the data used for index training to the learned index.
        """
        super().__init__()
        # to be defined in ``fit``
        self._index: faiss.Index | None = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "FaissNeighbors":
        _, dim = data.shape
        self._index = self._create_index(dim)
        self._index.train(data)  # type: ignore[reportCallIssue]

        if self.parameters.add_data_on_fit:
            # data might be added directly on fit, or using the ``add`` method
            self._index.add(data)  # type: ignore[reportCallIssue]

        return self

    @typecheck
    def add(self, data: Float[NumpyArray, "n d"]) -> "FaissNeighbors":
        self._index.add(data)  # type: ignore[reportCallIssue]
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], Float32[NumpyArray, "{n_neighbors}"]]:
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors)
        return idx.ravel(), dist.ravel()

    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "m {n_neighbors}"], Float32[NumpyArray, "m {n_neighbors}"]]:
        dist, idx = self._index.search(points, n_neighbors)  # type: ignore[reportCallIssue]
        return idx, dist

    def _create_index(self, dim: int) -> faiss.Index:
        index = self.parameters.index
        if isinstance(index, str):
            # if the index is a string, we treat it as an index factory
            return faiss.index_factory(dim, index)

        if isinstance(index, faiss.Index):
            # if the index is an existing index, simply use that
            return index

        # otherwise the index should be a callable returning an index
        return index(dim)
