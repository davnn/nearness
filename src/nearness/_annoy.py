from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from safecheck import Float, Int64, NumpyArray, Real, typecheck
from typing_extensions import Any, Iterable, Literal, overload

from ._base import NearestNeighbors
from ._base._helpers import load_index_from_temp_file, save_index_to_temp_file

__all__ = [
    "AnnoyNeighbors",
]


class AnnoyNeighbors(NearestNeighbors):
    available_metrics = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "euclidean",
        n_trees: int = 32,
        n_search_neighbors: int | None = None,
        random_seed: int | None = None,
        disk_build_path: str | Path | None = None,
        save_index_path: str | Path | None = None,
        load_index_path: str | Path | None = None,
        load_index_dim: int | None = None,
        prefault: bool = False,
    ) -> None:
        """Instantiate annoy nearest neighbors.

        :param metric: One of ["angular", "euclidean", "manhattan", "hamming", "dot"]
        :param n_trees: Builds a forest of ``n_trees`` trees. More trees gives higher precision when querying.
        :param n_search_neighbors: Inspect up to ``n_search_neighbors`` nodes, default is ``n_search_neighbors`` * n.
        :param random_seed: Initialize the random number generator with the given seed.
        :param disk_build_path: Build the index on disk given the ``disk_build_path``.
        :param save_index_path: Save the index on disk after build given ``save_index_path``.
        :param load_index_path: Loads (mmaps) an index from disk given ``load_index_path``, requires ``load_index_dim``.
        :param load_index_dim: Specify the dimension for a loaded index.
        :param prefault: If prefault is set to True, it will pre-read the entire file into memory (mmap MAP_POPULATE).
        """
        super().__init__()
        self._index: AnnoyIndex | None = None

        # mmap the raw annoy index if ``load_index_path`` is given
        if (path := self.parameters.load_index_path) is not None:
            # raise an exception if ``load_index_dim`` is not given
            if self.parameters.load_index_dim is None:
                msg = f"Cannot load index from path '{path}' if 'load_index_dim' is None."
                raise AssertionError(msg)

            index = AnnoyIndex(self.parameters.load_index_dim, self.parameters.metric)
            index.load(str(path))
            self._index = index
            self.__fitted__ = True

    @overload
    def fit(self, data: Iterable[Real[NumpyArray, "d"]]) -> "AnnoyNeighbors": ...

    @overload
    def fit(self, data: Real[NumpyArray, "n d"]) -> "AnnoyNeighbors": ...

    def fit(self, data):  # type: ignore[reportGeneralTypeIssues]
        n_samples, n_dim = data.shape
        self.parameters.load_index_dim = n_dim
        index = AnnoyIndex(n_dim, metric=self.parameters.metric)

        if (seed := self.parameters.random_seed) is not None:
            index.set_seed(seed)

        if (path := self.parameters.disk_build_path) is not None:
            index.on_disk_build(str(path))

        for i in range(n_samples):
            index.add_item(i, data[i].tolist())

        index.build(self.parameters.n_trees)

        if (path := self.parameters.save_index_path) is not None:
            index.save(str(path), prefault=self.parameters.prefault)

        self._index = index
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], Float[NumpyArray, "{n_neighbors}"]]:
        """No runtime type checks, because query is used in the ``query_batch`` loop."""
        idx, dist = self._index.get_nns_by_vector(
            point.tolist(),
            n_neighbors,
            self.parameters.n_search_neighbors,
            include_distances=True,
        )
        return np.array(idx, dtype=np.int64), np.array(dist, dtype=point.dtype)

    def query_idx(self, point: Float[NumpyArray, "d"], n_neighbors: int) -> Int64[NumpyArray, "{n_neighbors}"]:
        idx = self._index.get_nns_by_vector(
            point.tolist(),
            n_neighbors,
            self.parameters.n_search_neighbors,
        )
        return np.array(idx, dtype=np.int64)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Load the index using a temporary file."""
        super().__setstate__(state)
        if self._index is not None:
            index_bytes = state.pop("_index")
            self._index = AnnoyIndex(self.parameters.load_index_dim, self.parameters.metric)
            load_index_from_temp_file(index_bytes, load_fn=self._index.load, prefault=self.parameters.prefault)

    def __getstate__(self) -> dict[str, Any]:
        """Save the index to a temporary file and store the bytes of the file."""
        state = super().__getstate__()
        if self._index is not None:
            state["_index"] = save_index_to_temp_file(self._index.save, prefault=self.parameters.prefault)
        return state
