import warnings
from pathlib import Path

import usearch.index as usearch
from numpy import atleast_2d  # numpy is a dependecy of usearch, import should be safe
from safecheck import Float, Integer, NumpyArray, typecheck
from typing_extensions import Any

from ._base import NearestNeighbors
from ._base._helpers import IndexWrapper, save_index_to_temp_file

__all__ = ["UsearchIndex", "UsearchNeighbors"]


class UsearchIndex(IndexWrapper[usearch.Index]): ...


DEFAULT_INDEX = UsearchIndex(metric="l2sq")


class UsearchNeighbors(NearestNeighbors):
    """A simple wrapper around ``usearch.Index``.

    References
    ----------
        - https://github.com/unum-cloud/usearch

    """

    @typecheck
    def __init__(
        self,
        *,
        index: usearch.Index | UsearchIndex = DEFAULT_INDEX,
        exact_search: bool = False,
        copy_data: bool = True,
        threads_fit: int = 0,
        threads_search: int = 0,
        log_fit: bool = False,
        log_search: bool = False,
        progress_fit: usearch.ProgressCallback | None = None,
        progress_search: usearch.ProgressCallback | None = None,
        progress_save: usearch.ProgressCallback | None = None,
        progress_load: usearch.ProgressCallback | None = None,
        save_index_path: str | Path | None = None,
        load_index_path: str | Path | None = None,
        map_file_index: bool = False,
        add_data_on_fit: bool = True,
    ) -> None:
        """Instantiate Usearch nearest neighbors.

        :param index: A UsearchIndex wrapped index or a usearch.Index.
        :param exact_search: Bypass index and use bruteforce exact search.
        :param copy_data: Should the index store a copy of vectors.
        :param threads_fit: Optimal number of cores to use for index creation.
        :param threads_search: Optimal number of cores to use for index search.
        :param log_fit: Whether to print the progress bar on index creation.
        :param log_search: Whether to print the progress bar on index search.
        :param progress_fit: Callback to report stats of the index creation progress.
        :param progress_search: Callback to report stats of the index search progress.
        :param progress_save: Callback to report stats of the index save progress.
        :param progress_load: Callback to report stats of the index load progress.
        :param save_index_path: Save the index after index creation.
        :param load_index_path: Load an existing index directly on ``__init__``.
        :param map_file_index: Memory map an index after creation or on load.
        :param add_data_on_fit: Add data to the index directly on index creation.
        """
        super().__init__()
        # to be defined in ``fit``
        self._index: usearch.Index | None = None

        # load the index from file
        if (path := self.parameters.load_index_path) is not None:
            if save_index_path is not None:
                msg = f"Using 'load_index_path' ({path}), but 'save_index_path' is not None and is ignored."
                warnings.warn(msg, stacklevel=1)

            if not isinstance(self.parameters.index, usearch.Index):
                msg = f"Using 'load_index_path' ({path}), but no index given, using 'restore' to load."
                warnings.warn(msg, stacklevel=1)
            else:
                # set the index here, such that load uses it to load from file
                self._index = self.parameters.index

            # load index and manually set ``__fitted__``, such that fit validation is not enabled
            self._load_index(path)
            self.__fitted__ = True

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "UsearchNeighbors":
        _, dim = data.shape
        self._index = self._create_index(dim)

        if self.parameters.add_data_on_fit:
            # data might be added directly on fit, or using the ``add`` method
            self.add(data)

        if (path := self.parameters.save_index_path) is not None:
            self._index.save(
                path_or_buffer=path,
                progress=self.parameters.progress_save,
            )
            if self.parameters.map_file_index:
                self._index.reset()
                self._index.view(path)

        return self

    @typecheck
    def add(self, data: Float[NumpyArray, "n d"]) -> "UsearchNeighbors":
        self._index.add(
            keys=None,  # type: ignore[reportArgumentType]
            vectors=data,
            copy=self.parameters.copy_data,
            threads=self.parameters.threads_fit,
            log=self.parameters.log_fit,
            progress=self.parameters.progress_fit,
        )
        return self

    @typecheck
    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Integer[NumpyArray, "{n_neighbors}"], Float[NumpyArray, "{n_neighbors}"]]:
        match = self._search(point, n_neighbors)
        return match.keys, match.distances

    @typecheck
    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Integer[NumpyArray, "m {n_neighbors}"], Float[NumpyArray, "m {n_neighbors}"]]:
        match = self._search(points, n_neighbors)
        idxs, dists = match.keys, match.distances
        return tuple(atleast_2d(idxs, dists))  # type: ignore[reportReturnType]

    def _search(
        self,
        point_or_points: Float[NumpyArray, "d"] | Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> usearch.Matches | usearch.BatchMatches:
        return self._index.search(
            point_or_points,
            count=n_neighbors,
            threads=self.parameters.threads_search,
            exact=self.parameters.exact_search,
            log=self.parameters.log_search,
            progress=self.parameters.progress_search,
        )

    def _create_index(self, dim: int) -> usearch.Index:
        index = self.parameters.index
        # if the index is an existing index, use it, otherwise the index should be a callable returning an index
        return index if isinstance(index, usearch.Index) else index(ndim=dim)

    def _load_index(self, path: str | Path) -> None:
        if not Path(path).is_file():
            msg = f"Tried to load index from '{path}', but no file fount at path."
            raise AssertionError(msg)

        if isinstance(self._index, usearch.Index):
            load = self._index.view if self.parameters.map_file_index else self._index.load
            load(path, progress=self.parameters.progress_load)
        else:
            self._index = usearch.Index.restore(path, view=self.parameters.map_file_index)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Load the index using a temporary file."""
        super().__setstate__(state)
        if self._index is not None:
            index_bytes = state.pop("_index")
            self._index = usearch.Index.restore(index_bytes, view=False)
            return

        if path := self.parameters.load_index_path:
            self._load_index(path)
            return

        if (path := self.parameters.save_index_path) and self.is_fitted:
            self._load_index(path)
            return

    def __getstate__(self) -> dict[str, Any]:
        """Save the index to a temporary file and store the bytes of the file."""
        state = super().__getstate__()
        if self._index is not None:

            # index path is given, don't persist the index data
            if self.parameters.load_index_path is not None or self.parameters.save_index_path is not None:
                state["_index"] = None  # load using restore, don't persist the index
                return state

            # no index path given, index is in-memory, persist it
            state["_index"] = save_index_to_temp_file(self._index.save, progress=self.parameters.progress_save)
        return state
