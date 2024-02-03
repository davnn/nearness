from safecheck import Float64, Int64, NumpyArray, Real, typecheck
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors
from typing_extensions import Any, Literal

from ._base import NearestNeighbors


class SklearnNeighbors(NearestNeighbors):
    """CPU-based nearest neighbors algorithm based on scikit-learn. Note: The distances and indices are sorted!."""

    available_algorithms = Literal["auto", "brute", "ball_tree", "kd_tree"]

    @typecheck
    def __init__(
        self,
        *,
        algorithm: available_algorithms = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",  # dynamically checked
        p: int = 2,
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__()
        # to be set in ``fit``
        self._model: None | SklearnNearestNeighbors = None

    @typecheck
    def fit(self, data: Real[NumpyArray, "n d"]) -> "SklearnNeighbors":
        self._model = SklearnNearestNeighbors(
            algorithm=self.parameters.algorithm,
            leaf_size=self.parameters.leaf_size,
            metric=self.parameters.metric,
            p=self.parameters.p,
            metric_params=self.parameters.metric_params,
            n_jobs=self.parameters.n_jobs,
        )
        self._model.fit(data)
        return self

    def query(
        self,
        point: Real[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], Float64[NumpyArray, "{n_neighbors}"]]:
        """CPU-based nearest neighbors algorithm based on scikit-learn. Note: The distances and indices are sorted!."""
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors=n_neighbors)
        return idx.ravel(), dist.ravel()

    def query_batch(
        self,
        points: Real[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "m {n_neighbors}"], Float64[NumpyArray, "m {n_neighbors}"]]:
        dist, idx = self._model.kneighbors(points, n_neighbors=n_neighbors)
        return idx, dist
