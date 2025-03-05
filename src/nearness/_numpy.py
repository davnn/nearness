import numpy as np
from safecheck import Float, Int64, NumpyArray, typecheck
from typing_extensions import Literal

from ._base import NearestNeighbors


def cdist_minkowski(
    data: Float[NumpyArray, "n d"],
    query: Float[NumpyArray, "m d"],
    p: int | float = 2,
) -> Float[NumpyArray, "m n"]:
    """Reference implementation of Euclidean distance with NumPy.

    :param data: Matrix of input data points.
    :param query: Matrix of query data points.
    :param p: The power parameter for the Minkowski distance metric.
    :return: Matrix of distances between query and data.
    """
    diff = query[:, np.newaxis, :] - data[np.newaxis, :, :]
    return np.linalg.norm(diff, ord=p, axis=-1, keepdims=False)


def cdist_euclidean_mm(
    data: Float[NumpyArray, "n d"],
    query: Float[NumpyArray, "m d"],
) -> Float[NumpyArray, "m n"]:
    """Probably the fastest possible implementation of Euclidean distance with NumPy.

    :param data: Matrix of input data points.
    :param query: Matrix of query data points.
    :return: Matrix of distances between query and data.
    """
    # Calculate the squared differences raised to the power of p
    squared_diff_p = (
        np.einsum("ij,ij->i", data, data)[np.newaxis, :]
        - 2.0 * np.dot(query, data.T)
        + np.einsum("ij,ij->i", query, query)[:, np.newaxis]
    )

    # Ensure that negative values close to zero are treated as zeros due to potential numerical errors
    np.maximum(squared_diff_p, 0.0, out=squared_diff_p)
    return np.sqrt(squared_diff_p)


def min_k(arr: Float[NumpyArray, "m n"], k: int) -> tuple[Int64[NumpyArray, "m {k}"], Float[NumpyArray, "m {k}"]]:
    """Similar to torch.topk, but only implemented for matrices and returning the smallest (min) items.

    :param arr: Matrix of data points (distances).
    :param k: Number of smallest elements to determine.
    """
    indices = np.argpartition(arr, kth=range(k), axis=1)[:, :k]
    values = np.take_along_axis(arr, indices, axis=1)
    return indices, values


class NumpyNeighbors(NearestNeighbors):
    """Numpy-based exact nearest neighbors implementation."""

    available_metrics = Literal["minkowski"]
    compute_modes = Literal["use_mm_for_euclid_dist", "donot_use_mm_for_euclid_dist"]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "minkowski",
        p: int = 2,
        compute_mode: compute_modes = "use_mm_for_euclid_dist",
    ) -> None:
        """Instantiate Numpy nearest neighbors.

        :param metric: Only "minkowski" is supported currently.
        :param p: Parameter that defines the specific p-norm used.
        :param compute_mode: Use matrix multiple when ``p=2`` and mode is "use_mm_for_euclid_dist".
        """
        super().__init__()
        # to be defined in ``fit``
        self._data: Float[NumpyArray, "n d"] | None = None

    @typecheck
    def fit(self, data: Float[NumpyArray, "n d"]) -> "NumpyNeighbors":
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
        if self.parameters.p == 2 and self.parameters.compute_mode == "use_mm_for_euclid_dist":
            distance = cdist_euclidean_mm(self._data, points)  # type: ignore[reportGeneralTypeIssues]
        else:
            distance = cdist_minkowski(self._data, points, p=self.parameters.p)  # type: ignore[reportGeneralTypeIssues]

        return min_k(distance, k=n_neighbors)
