import logging

import faiss
from autofaiss import build_index
from safecheck import Float, Float32, Int64, NumpyArray, typecheck

from ._base import NearestNeighbors


class AutoFaissNeighbors(NearestNeighbors):
    """Similar to ``FaissNeighbors``, but the definition of the index is automated.

    Given the constraints provided in the arguments, ``autofaiss`` chooses an appropriate index.
    """

    @typecheck
    def __init__(
        self,
        *,
        save_on_disk: bool = False,
        pre_load_index: bool = False,
        pre_load_using_mmap: bool = False,
        index_path: str | None = None,
        index_infos_path: str | None = None,
        ids_path: str | None = None,
        file_format: str = "npy",
        embedding_column_name: str = "embedding",
        id_columns: list[str] | None = None,
        index_key: str | None = None,
        index_param: str | None = None,
        max_index_query_time_ms: int | float = 10.0,
        max_index_memory_usage: str = "16G",
        min_nearest_neighbors_to_retrieve: int = 20,
        current_memory_available: str = "32G",
        use_gpu: bool = False,
        metric_type: str = "ip",
        nb_cores: int | None = None,
        make_direct_map: bool = False,
        should_be_memory_mappable: bool = False,
        distributed: str | None = None,
        temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
        verbose: int = logging.INFO,
        nb_indices_to_keep: int = 1,
    ) -> None:
        super().__init__()
        # to be defined in ``fit``
        self._index = None
        self._infos = None

        if pre_load_index:
            if pre_load_using_mmap:
                self._index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            else:
                self._index = faiss.read_index(index_path)
            self.is_fitted = True

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "AutoFaissNeighbors":
        self._index, self._infos = build_index(
            data,
            save_on_disk=self.parameters.save_on_disk,
            index_infos_path=self.parameters.index_infos_path,
            ids_path=self.parameters.ids_path,
            file_format=self.parameters.file_format,
            embedding_column_name=self.parameters.embedding_column_name,
            id_columns=self.parameters.id_columns,
            index_key=self.parameters.index_key,
            index_param=self.parameters.index_param,
            max_index_query_time_ms=self.parameters.max_index_query_time_ms,
            max_index_memory_usage=self.parameters.max_index_memory_usage,
            min_nearest_neighbors_to_retrieve=self.parameters.min_nearest_neighbors_to_retrieve,
            current_memory_available=self.parameters.current_memory_available,
            use_gpu=self.parameters.use_gpu,
            metric_type=self.parameters.metric_type,
            nb_cores=self.parameters.nb_cores,
            make_direct_map=self.parameters.make_direct_map,
            should_be_memory_mappable=self.parameters.should_be_memory_mappable,
            distributed=self.parameters.distributed,
            temporary_indices_folder=self.parameters.temporary_indices_folder,
            verbose=self.parameters.verbose,
            nb_indices_to_keep=self.parameters.nb_indices_to_keep,
        )
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
