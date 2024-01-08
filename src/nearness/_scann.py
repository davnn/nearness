from dataclasses import asdict, dataclass
from pathlib import Path

import scann
from safecheck import Float, Float32, NumpyArray, UInt32, typecheck
from typing_extensions import Literal, NotRequired, TypedDict, get_type_hints

from ._base import NearestNeighbors


@dataclass
class ScannTreeConfig:
    num_leaves: int = 1000
    num_leaves_to_search: int = 10
    training_sample_size: int = 100000
    min_partition_size: int = 50
    training_iterations: int = 10
    spherical: bool = False
    quantize_centroids: bool = False
    random_init: bool = True


@dataclass
class ScannHashingConfig:
    dimensions_per_block: int = 2
    anisotropic_quantization_threshold: float = 0.2
    training_sample_size: int = 100000
    min_cluster_size: int = 100
    hash_type: str = "lut16"
    training_iterations: int = 10


@dataclass
class ScannBruteForceConfig:
    quantize: bool = False


@dataclass
class ScannReorderConfig:
    reordering_num_neighbors: int = 100
    quantize: bool = False


@dataclass
class ScannSearchConfig:
    pre_reorder_num_neighbors: int | None = None
    leaves_to_search: int | None = None


def typedict(data_class: type) -> type[TypedDict]:  # type: ignore[reportGeneralTypeIssues]
    """Convert the dataclass to a typed dictionary.

    All dictionary items are set to ``NotRequired`` because we have default initializers
    for all config items where ``typedict`` is used.

    :param data_class: Configuration data class with default values.
    :return: TypedDict representation of the dataclass.
    """
    field_types: dict[str, type] = {
        k: NotRequired[t] for k, t in get_type_hints(data_class).items()  # type: ignore[reportGeneralTypeIssues]
    }
    return TypedDict(f"{data_class.__name__}TypedDict", field_types)  # type: ignore[reportGeneralTypeIssues]


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
        use_reorder: bool = False,
        use_bruteforce: bool = False,
        search_parallel: bool = False,
        tree_config: ScannTreeConfig | typedict(ScannTreeConfig) | None = None,  # type: ignore[reportGeneralTypeIssues]
        bruteforce_config: ScannBruteForceConfig | typedict(ScannBruteForceConfig) | None = None,  # type: ignore[reportGeneralTypeIssues]
        hashing_config: ScannHashingConfig | typedict(ScannHashingConfig) | None = None,  # type: ignore[reportGeneralTypeIssues]
        reorder_config: ScannReorderConfig | typedict(ScannReorderConfig) | None = None,  # type: ignore[reportGeneralTypeIssues]
        search_config: ScannSearchConfig | typedict(ScannSearchConfig) | None = None,  # type: ignore[reportGeneralTypeIssues]
    ) -> None:
        super().__init__()
        # safely initialize the mutable config parameters
        to_config = lambda c, v: c() if v is None else c(**v) if isinstance(v, dict) else v
        self.parameters.tree_config = to_config(ScannTreeConfig, tree_config)
        self.parameters.bruteforce_config = to_config(ScannBruteForceConfig, bruteforce_config)
        self.parameters.hashing_config = to_config(ScannHashingConfig, hashing_config)
        self.parameters.reorder_config = to_config(ScannReorderConfig, reorder_config)
        self.parameters.search_config = to_config(ScannSearchConfig, search_config)

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
            searcher.tree(**asdict(parameters.tree_config))

        # provide the score config
        getattr(searcher, score_method)(**asdict(score_config))

        # provide the reorder config
        if self.parameters.use_reorder:
            searcher.reorder(**asdict(parameters.reorder_config))

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
            **asdict(self.parameters.search_config),
        )
        return idx, dist

    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[UInt32[NumpyArray, "m {n_neighbors}"], Float32[NumpyArray, "m {n_neighbors}"]]:
        search_method = "search_batched_parallel" if self.parameters.search_parallel else "search_batched"
        idx, dist = getattr(self._index, search_method)(
            points,
            final_num_neighbors=n_neighbors,
            **asdict(self.parameters.search_config),
        )
        return idx, dist

    def save(self, file: str | Path) -> None:
        self._assert_is_directory(file)
        self._index.serialize(str(file))

    def load(self, file: str | Path) -> "ScannNeighbors":
        self._assert_is_directory(file)
        searcher = scann.scann_ops_pybind.load_searcher(str(file))
        self._index = searcher
        self.__fitted__ = True
        return self

    @staticmethod
    def _assert_is_directory(file: str | Path) -> None:
        if not Path(file).is_dir():
            msg = f"ScaNN requires an artifacts directory for storage, but found no directory at '{file}'."
            raise AssertionError(msg)
