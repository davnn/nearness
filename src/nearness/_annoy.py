import os
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from annoy import AnnoyIndex
from safecheck import Float, Int64, NumpyArray, Real, typecheck
from typing_extensions import Any, Iterable, Literal, overload

from ._base import NearestNeighbors

IS_WINDOWS = os.name == "nt"

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
        n_search_neighbors: int = 128,
        random_seed: int | None = None,
        disk_build_path: str | Path | None = None,
        save_index_path: str | Path | None = None,
        load_index_path: str | Path | None = None,
        load_index_dim: int | None = None,
        prefault: bool = False,
    ) -> None:
        super().__init__()
        self._index: AnnoyIndex | None = None

        # mmap the raw annoy index
        if (path := self.parameters.load_index_path) is not None:
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
            # cannot delete the file on windows, ``delete_on_close`` is only available with Python >= 3.12
            with tempfile.NamedTemporaryFile("wb", delete=not IS_WINDOWS) as file:
                file.write(index_bytes)
                file.flush()  # otherwise errors with 'Inappropriate ioctl for device'

                # it's not allowed to re-open the file on windows
                # https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
                if IS_WINDOWS:
                    file.close()

                # open the file and load the index
                self._index.load(file.name, prefault=self.parameters.prefault)

    def __getstate__(self) -> dict[str, Any]:
        """Save the index to a temporary file and store the bytes of the file."""
        state = super().__getstate__()
        if self._index is not None:
            # there might be cleanup errors in multiprocessing setting on windows
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=IS_WINDOWS) as tmp_dir:
                file_path = Path(tmp_dir) / uuid4().hex
                self._index.save(str(file_path), prefault=self.parameters.prefault)
                state["_index"] = file_path.read_bytes()
        return state
