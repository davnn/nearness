from safecheck import Float, Int64, NumpyArray, typecheck
from scipy.spatial.distance import cdist
from typing_extensions import Literal, TypedDict

from ._base import NearestNeighbors
from ._numpy import min_k


class ScipyNeighbors(NearestNeighbors):
    """CPU-based nearest neighbors algorithm based on scikit-learn. Note: The distances and indices are sorted!."""

    available_metrics = Literal[
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulczynski1",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]

    class MetricArgs(TypedDict):
        p: float
        w: NumpyArray
        V: NumpyArray
        VI: NumpyArray
        out: NumpyArray

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "euclidean",
        metric_args: MetricArgs | None = None,
    ) -> None:
        super().__init__()
        if metric_args is None:
            self.parameters.metric_args = {}

        # to be defined in ``fit``
        self._data: Float[NumpyArray, "n d"] | None = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "ScipyNeighbors":
        self._data = data
        return self

    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], Float[NumpyArray, "{n_neighbors}"]]:
        """CPU-based nearest neighbors algorithm based on scikit-learn. Note: The distances and indices are sorted!."""
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors)
        return idx.ravel(), dist.ravel()

    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "m {n_neighbors}"], Float[NumpyArray, "m {n_neighbors}"]]:
        distance = cdist(
            points,
            self._data,  # type: ignore[reportGeneralTypeIssues]
            metric=self.parameters.metric,
            **self.parameters.metric_args,
        )
        idx, dist = min_k(distance, k=n_neighbors)
        return idx, dist
