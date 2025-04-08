import jax
import jax.numpy as jnp
import numpy as np
from safecheck import Float, Float32, Int32, JaxArray, NumpyArray, is_instance, typecheck
from typing_extensions import Literal, overload

from ._base import NearestNeighbors


def cdist_minkowski(
    data: Float[NumpyArray | JaxArray, "n d"],
    query: Float[NumpyArray | JaxArray, "m d"],
    p: int | float = 2,
) -> Float[NumpyArray | JaxArray, "m n"]:
    """Reference implementation of Euclidean distance with Jax.

    :param data: Matrix of input data points.
    :param query: Matrix of query data points.
    :param p: The power parameter for the Minkowski distance metric.
    :return: Matrix of distances between query and data.
    """
    diff = jnp.expand_dims(query, 1) - jnp.expand_dims(data, 0)
    return jnp.linalg.norm(diff, ord=p, axis=-1, keepdims=False)


def cdist_euclidean_mm(
    data: Float[NumpyArray | JaxArray, "n d"],
    query: Float[NumpyArray | JaxArray, "m d"],
) -> Float[NumpyArray | JaxArray, "m n"]:
    """Probably the fastest possible implementation of Euclidean distance with NumPy.

    :param data: Matrix of input data points.
    :param query: Matrix of query data points.
    :return: Matrix of distances between query and data.
    """
    # Calculate the squared differences raised to the power of p
    squared_diff_p = (
        jnp.expand_dims(jnp.einsum("ij,ij->i", data, data), 0)
        - 2.0 * jnp.dot(query, data.T)
        + jnp.expand_dims(jnp.einsum("ij,ij->i", query, query), 1)
    )

    # Ensure that negative values close to zero are treated as zeros due to potential numerical errors
    return jnp.sqrt(jnp.maximum(squared_diff_p, 0.0))


class JaxNeighbors(NearestNeighbors):
    """Jax-based exact nearest neighbors implementation with option for inexact neighbors sorting."""

    available_metrics = Literal["minkowski"]
    compute_modes = Literal["use_mm_for_euclid_dist", "donot_use_mm_for_euclid_dist"]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "minkowski",
        p: int = 2,
        compute_mode: compute_modes = "use_mm_for_euclid_dist",
        approximate_recall_target: float = 0.95,
    ) -> None:
        """Instantiate Jax nearest neighbors.

        :param metric: Only "minkowski" is supported currently.
        :param p: Parameter that defines the specific p-norm used.
        :param compute_mode: Use matrix multiple when ``p=2`` and mode is "use_mm_for_euclid_dist".
        :param approximate_recall_target: Recall target for nearest neighbors sorting.
        """
        super().__init__()
        # to be defined in ``fit``
        self._data: Float[JaxArray, "n d"] | None = None

    @typecheck
    def fit(self, data: Float[NumpyArray | JaxArray, "n d"]) -> "JaxNeighbors":
        self._data = jnp.asarray(data)
        return self

    @overload
    def query(
        self,
        point: Float[JaxArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int32[JaxArray, "{n_neighbors}"], Float32[JaxArray, "{n_neighbors}"]]: ...

    @overload
    def query(
        self,
        point: Float[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int32[NumpyArray, "{n_neighbors}"], Float32[NumpyArray, "{n_neighbors}"]]: ...

    def query(self, point, n_neighbors):  # type: ignore[reportGeneralTypeIssues]
        """CPU-based nearest neighbors algorithm based on scikit-learn. Note: The distances and indices are sorted!."""
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors)
        return idx.ravel(), dist.ravel()

    @overload
    def query_batch(
        self,
        points: Float[JaxArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int32[JaxArray, "m {n_neighbors}"], Float32[JaxArray, "m {n_neighbors}"]]: ...

    @overload
    def query_batch(
        self,
        points: Float[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int32[NumpyArray, "m {n_neighbors}"], Float32[NumpyArray, "m {n_neighbors}"]]: ...

    def query_batch(self, points, n_neighbors):  # type: ignore[reportGeneralTypeIssues]
        # the data is implicitly casted to a jax array
        is_numpy = is_instance(points, NumpyArray)
        if self.parameters.p == 2 and self.parameters.compute_mode == "use_mm_for_euclid_dist":
            distance = cdist_euclidean_mm(self._data, points)  # type: ignore[reportGeneralTypeIssues]
        else:
            distance = cdist_minkowski(self._data, points, p=self.parameters.p)  # type: ignore[reportGeneralTypeIssues]

        dist, idx = jax.lax.approx_min_k(
            distance,  # type: ignore[reportGeneralTypeIssues]
            n_neighbors,
            recall_target=self.parameters.approximate_recall_target,
        )
        # ``np.asarray`` should be the best option to ensure no copy is made, https://github.com/google/jax/issues/1961
        if is_numpy:
            return np.asarray(idx), np.asarray(dist)

        return idx, dist
