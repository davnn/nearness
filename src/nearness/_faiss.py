import faiss
import numpy as np
from safecheck import Float, Float32, Int64, NumpyArray, typecheck
from typing_extensions import Any, Protocol, runtime_checkable

from ._base import NearestNeighbors


@runtime_checkable
class FaissIndexFactory(Protocol):
    def __call__(self, dim: int) -> faiss.Index:
        ...


class FaissNeighbors(NearestNeighbors):
    """A fairly simple wrapper around ``faiss.Index``.

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
        index_or_factory: str | FaissIndexFactory | faiss.Index = "Flat",
        add_data_on_fit: bool = True,
        sample_train_points: int | float | None = None,
        sample_with_replacement: bool = False,
        rng: Any = None,  # types are validated by default_rng
    ) -> None:
        super().__init__()
        self.parameters.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        # to be defined in ``fit``
        self._index: faiss.Index | None = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "FaissNeighbors":
        n_samples, dim = data.shape
        self._index = self._create_index(dim)
        if (n_train := self.parameters.sample_train_points) is not None:
            if isinstance(n_train, float):
                # we make sure float is 0 < float < 1 in ``__init__``
                n_train = int(n_train * n_samples)

            sample = self.parameters.rng.choice(data, size=n_train, replace=self.parameters.sample_with_replacement)
            self._index.train(sample)  # type: ignore[reportGeneralTypeIssues]
        else:
            self._index.train(data)  # type: ignore[reportGeneralTypeIssues]

        if self.parameters.add_data_on_fit:
            # data might be added directly on fit, or using the ``add`` method
            self._index.add(data)  # type: ignore[reportGeneralTypeIssues]

        return self

    @typecheck
    def add(self, data: Float[NumpyArray, "n d"]) -> "FaissNeighbors":
        self._index.add(data)  # type: ignore[reportGeneralTypeIssues]
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
        dist, idx = self._index.search(points, n_neighbors)  # type: ignore[reportGeneralTypeIssues]
        return idx, dist

    def _create_index(self, dim: int) -> faiss.Index:
        index = self.parameters.index_or_factory
        if isinstance(index, str):
            # if the index is a string, we treat it as an index factory
            return faiss.index_factory(dim, index)

        if isinstance(index, faiss.Index):
            # if the index is an existing index, simply use that
            return index

        # otherwise the index should be a callable returning an index
        return index(dim)
