from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from safecheck import Float, Int64, Is, NumpyArray, Real, typecheck
from typing_extensions import Annotated, Iterable, Literal, overload

from ._base import NearestNeighbors


class AnnoyNeighbors(NearestNeighbors):
    available_metrics = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]
    positive_int = Annotated[int, Is[lambda i: i > 0]]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "euclidean",
        n_trees: positive_int = 32,
        n_search_neighbors: positive_int = 128,
        random_seed: int | None = None,
        on_disk_build: str | None = None,
        load_dim: positive_int | None = None,
        prefault: bool = False,
    ) -> None:
        super().__init__()
        self._model = None

    @overload
    def fit(self, data: Iterable[Real[NumpyArray, "d"]]) -> "AnnoyNeighbors":
        ...

    @overload
    def fit(self, data: Real[NumpyArray, "n d"]) -> "AnnoyNeighbors":
        ...

    def fit(self, data):  # type: ignore[reportGeneralTypeIssues]
        n_samples, n_dim = data.shape
        self.parameters.load_dim = n_dim
        model = AnnoyIndex(n_dim, metric=self.parameters.metric)

        if (seed := self.parameters.random_seed) is not None:
            model.set_seed(seed)

        if (path := self.parameters.on_disk_build) is not None:
            model.on_disk_build(path)

        for i in range(n_samples):
            model.add_item(i, data[i].tolist())

        model.build(self.parameters.n_trees)
        self._model = model
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], Float[NumpyArray, "{n_neighbors}"]]:
        """No runtime type checks, because query is used in the ``query_batch`` loop."""
        idx, dist = self._model.get_nns_by_vector(
            point.tolist(),
            n_neighbors,
            self.parameters.n_search_neighbors,
            include_distances=True,
        )
        return np.array(idx, dtype=np.int64), np.array(dist, dtype=point.dtype)

    def query_idx(self, point: Float[NumpyArray, "d"], n_neighbors: int) -> Int64[NumpyArray, "{n_neighbors}"]:
        idx = self._model.get_nns_by_vector(
            point.tolist(),
            n_neighbors,
            self.parameters.n_search_neighbors,
        )
        return np.array(idx, dtype=np.int64)

    def save(self, file: str | Path) -> None:
        self._model.save(str(file), prefault=self.parameters.prefault)

    def load(self, file: str | Path) -> "NearestNeighbors":
        model = AnnoyIndex(self.parameters.load_dim, metric=self.parameters.metric)
        model.load(str(file), prefault=self.parameters.prefault)
        self._model = model
        self.__fitted__ = True
        return self
