import tempfile
from pathlib import Path

import scann
from safecheck import Float, Float32, NumpyArray, UInt32, typecheck
from scann.scann_ops.py import scann_ops_pybind_backcompat
from typing_extensions import Any, Literal, TypedDict

from ._base import NearestNeighbors


class ScannTreeConfig(TypedDict):
    num_leaves: int
    num_leaves_to_search: int
    training_sample_size: int
    min_partition_size: int
    training_iterations: int
    spherical: bool
    quantize_centroids: bool
    random_init: bool

    @staticmethod  # type: ignore[reportGeneralTypeIssues]
    def create() -> "ScannTreeConfig":
        return ScannTreeConfig(
            num_leaves=1000,
            num_leaves_to_search=10,
            training_sample_size=100000,
            min_partition_size=50,
            training_iterations=10,
            spherical=False,
            quantize_centroids=False,
            random_init=True,
        )


class ScannHashingConfig(TypedDict):
    dimensions_per_block: int
    anisotropic_quantization_threshold: float
    training_sample_size: int
    min_cluster_size: int
    hash_type: str
    training_iterations: int

    @staticmethod  # type: ignore[reportGeneralTypeIssues]
    def create() -> "ScannHashingConfig":
        return ScannHashingConfig(
            dimensions_per_block=2,
            anisotropic_quantization_threshold=0.2,
            training_sample_size=100000,
            min_cluster_size=100,
            hash_type="lut16",
            training_iterations=10,
        )


class ScannBruteForceConfig(TypedDict):
    quantize: bool

    @staticmethod  # type: ignore[reportGeneralTypeIssues]
    def create() -> "ScannBruteForceConfig":
        return ScannBruteForceConfig(quantize=False)


class ScannReorderConfig(TypedDict):
    reordering_num_neighbors: int
    quantize: bool

    @staticmethod  # type: ignore[reportGeneralTypeIssues]
    def create() -> "ScannReorderConfig":
        return ScannReorderConfig(
            reordering_num_neighbors=100,
            quantize=False,
        )


class ScannSearchConfig(TypedDict):
    pre_reorder_num_neighbors: int | None
    leaves_to_search: int | None

    @staticmethod  # type: ignore[reportGeneralTypeIssues]
    def create() -> "ScannSearchConfig":
        return ScannSearchConfig(
            pre_reorder_num_neighbors=None,
            leaves_to_search=None,
        )


class ScannNeighbors(NearestNeighbors):
    """ScaNN performs search in three phases as described in the following.

    1. Partitioning (optional): ScaNN partitions the dataset during training time, and at query time selects the top
    partitions to pass onto the scoring stage.
    2. Scoring: ScaNN computes the distances from the query to all datapoints in the dataset
    (if partitioning isn't enabled) or all datapoints in a partition to search (if partitioning is enabled).
    These distances aren't necessarily exact, but brute-force scoring can be enabled.
    3. Rescoring (optional): ScaNN takes the best k' distances from scoring and re-computes these distances more
    accurately. From these k' re-computed distances the top k are selected.

    Source: https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
    """

    available_metrics = Literal["dot_product", "squared_l2"]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "squared_l2",
        n_neighbors: int = 1,
        n_training_threads: int = 0,
        use_tree: bool = True,
        use_bruteforce: bool = False,
        use_reorder: bool = False,
        search_parallel: bool = False,
        tree_config: ScannTreeConfig | None = None,
        bruteforce_config: ScannBruteForceConfig | None = None,
        hashing_config: ScannHashingConfig | None = None,
        reorder_config: ScannReorderConfig | None = None,
        search_config: ScannSearchConfig | None = None,
    ) -> None:
        """Instantiate Scann nearest neighbors.

        :param metric: One of ["dot_product", "squared_l2"].
        :param n_neighbors: Number of neighbors specified for index creation (overriden on query).
        :param n_training_threads: Number of threads used for index creation.
        :param use_tree: Use tree for data partitioning.
        :param use_bruteforce: Use bruteforce approach for scoring, otherwise use asymmetric hashing (AH).
        :param use_reorder: Use rescoring of results, highly recommended if AH scoring is used.
        :param search_parallel: Perform batched searches in parallel.
        :param tree_config: Configuration parameters for tree partitioning.
        :param bruteforce_config: Configuration parameters for bruteforce search.
        :param hashing_config: Configuration parameters for asymmetric hashing search.
        :param reorder_config: Configuration parameters for score reordering (rescoring).
        :param search_config: Configuration parameters for searches.
        """
        super().__init__()
        # safely initialize the mutable config parameters
        to_config = lambda c, v: c.create() | v if isinstance(v, dict) else c.create()
        self.parameters.tree_config = to_config(ScannTreeConfig, tree_config)
        self.parameters.bruteforce_config = to_config(ScannBruteForceConfig, bruteforce_config)
        self.parameters.hashing_config = to_config(ScannHashingConfig, hashing_config)
        self.parameters.reorder_config = to_config(ScannReorderConfig, reorder_config)
        self.parameters.search_config = to_config(ScannSearchConfig, search_config)
        # the c++ search functions only allow integer inputs using default values when -1 is given, this was previously
        # transformed from None to -1 directly in the python wrapper, but is missing in 1.3.5 for ``searcher.search``.
        self.parameters.search_config = {k: -1 if v is None else v for k, v in self.parameters.search_config.items()}

        # to be defined in ``fit``
        self._index = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "ScannNeighbors":
        parameters = self.parameters
        score_config = parameters.bruteforce_config if parameters.use_bruteforce else parameters.hashing_config
        score_method = "score_brute_force" if parameters.use_bruteforce else "score_ah"

        # set up the builder
        searcher = scann.scann_ops_pybind.builder(
            db=data,
            num_neighbors=parameters.n_neighbors,
            distance_measure=parameters.metric,
        )

        # set number of threads
        searcher.set_n_training_threads(self.parameters.n_training_threads)

        # provide the tree config
        if self.parameters.use_tree:
            searcher.tree(**parameters.tree_config)

        # provide the score config
        getattr(searcher, score_method)(**score_config)

        # provide the reorder config
        if self.parameters.use_reorder:
            searcher.reorder(**parameters.reorder_config)

        self._index = searcher.build()
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[UInt32[NumpyArray, "{n_neighbors}"], Float32[NumpyArray, "{n_neighbors}"]]:
        idx, dist = self._index.search(
            point,
            final_num_neighbors=n_neighbors,
            **self.parameters.search_config,
        )
        return idx, dist  # type: ignore[reportReturnType]

    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[UInt32[NumpyArray, "m {n_neighbors}"], Float32[NumpyArray, "m {n_neighbors}"]]:
        search_method = "search_batched_parallel" if self.parameters.search_parallel else "search_batched"
        idx, dist = getattr(self._index, search_method)(
            points,
            final_num_neighbors=n_neighbors,
            **self.parameters.search_config,
        )
        return idx, dist

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Save the bytes to files and read the index from the temporary files."""
        super().__setstate__(state)
        if self._index is not None:
            index_data = state.pop("_index")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                for name, data in index_data.items():
                    (tmp_path / name).write_bytes(data)

                # override the scann_assets.pbtxt file in the current directory, otherwise the paths to the
                # assets include the old temporary directory prefixes
                scann_ops_pybind_backcompat.populate_and_save_assets_proto(tmp_dir)
                self._index = scann.scann_ops_pybind.load_searcher(tmp_dir)

    def __getstate__(self) -> dict[str, Any]:
        """Save the index to temporary directory and store the resulting bytes in a dictionary to re-build the index."""
        state = super().__getstate__()
        if self._index is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                self._index.serialize(tmp_dir)
                tmp_path = Path(tmp_dir)
                files = tmp_path.glob("*")
                data = {file.name: file.read_bytes() for file in files}
                state["_index"] = data

        return state
