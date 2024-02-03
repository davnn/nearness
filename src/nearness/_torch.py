# type: ignore[reportGeneralTypeIssues,reportIncompatibleMethodOverride]
# It appears to be impossible to correctly type the overrides without bloat.


import torch
from safecheck import AbstractDtype, Int64, NumpyArray, TorchArray, typecheck
from typing_extensions import Literal, overload

from ._base import NearestNeighbors


class SupportedFloat(AbstractDtype):
    dtypes = ["float32", "float64"]  # noqa: RUF012


class TorchNeighbors(NearestNeighbors):
    available_metrics = Literal["minkowski"]
    compute_modes = Literal[
        "use_mm_for_euclid_dist",
        "donot_use_mm_for_euclid_dist",
        "use_mm_for_euclid_dist_if_necessary",
    ]

    @typecheck
    def __init__(
        self,
        *,
        metric: available_metrics = "minkowski",
        p: int = 2,
        compute_mode: compute_modes = "use_mm_for_euclid_dist",
        force_dtype: torch.dtype | None = None,
        force_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        # to be defined in ``fit``
        self._data: SupportedFloat[TorchArray, "n d"] | None = None

    @typecheck
    def fit(self, data: SupportedFloat[NumpyArray | TorchArray, "n d"]) -> "TorchNeighbors":
        self._data = torch.as_tensor(data, device=self.parameters.force_device, dtype=self.parameters.force_dtype)
        return self

    @overload
    def query(
        self,
        point: SupportedFloat[NumpyArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "{n_neighbors}"], SupportedFloat[NumpyArray, "{n_neighbors}"]]: ...

    @overload
    def query(
        self,
        point: SupportedFloat[TorchArray, "d"],
        n_neighbors: int,
    ) -> tuple[Int64[TorchArray, "{n_neighbors}"], SupportedFloat[TorchArray, "{n_neighbors}"]]: ...

    def query(self, point, n_neighbors):
        idx, dist = self.query_batch(point.reshape(1, -1), n_neighbors)
        return idx.ravel(), dist.ravel()

    @overload
    def query_batch(
        self,
        points: SupportedFloat[NumpyArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int64[NumpyArray, "m {n_neighbors}"], SupportedFloat[NumpyArray, "m {n_neighbors}"]]: ...

    @overload
    def query_batch(
        self,
        points: SupportedFloat[TorchArray, "m d"],
        n_neighbors: int,
    ) -> tuple[Int64[TorchArray, "m {n_neighbors}"], SupportedFloat[TorchArray, "m {n_neighbors}"]]: ...

    def query_batch(self, points, n_neighbors):
        is_numpy = isinstance(points, NumpyArray)
        points = torch.as_tensor(points, dtype=self._data.dtype, device=self._data.device)
        dist = torch.cdist(points, self._data, p=self.parameters.p, compute_mode=self.parameters.compute_mode)
        dist, idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False)

        if is_numpy:
            return idx.cpu().numpy(), dist.cpu().numpy()

        return idx, dist
