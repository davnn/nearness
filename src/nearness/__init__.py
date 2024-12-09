"""Entrypoint to the library exposing all available algorithms (if they are importable).

An algorithm can only be imported if the required libraries are installed. For example,
importing ``AnnoyNeighbors`` requires the ``annoy`` package to be available.

The additional libraries can be installed using nearness' package extras. For example,
to install nearness with ``annoy``, run ``pip install nearness[annoy]`` or install ``annoy``
by other means, for example, using ``conda install conda-forge::python-annoy``.

Installing nearness without any extras exposes only the abstract base class ``NearestNeighbors``
to define your own nearest neighbors algorithm. However, if you already have installed a
package that would otherwise be included as an extra dependency, you can use the corresponding
algorithm. For example, if you have installed ``nearness`` and ``numpy``, the algorithm
``NumpyNeighbors`` is exposed.
"""

from ._base import ExperimentalWarning, NearestNeighbors, config

__all__ = [
    "ExperimentalWarning",
    "NearestNeighbors",
    "config",
]


def get_version() -> str:
    """Return the package version or "unknown" if no version can be found."""
    from importlib import metadata

    try:
        return metadata.version(__name__)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "no-version-found-in-package-metadata"


try:
    from ._annoy import AnnoyNeighbors

    __all__ += ["AnnoyNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._autofaiss import AutoFaissNeighbors

    __all__ += ["AutoFaissNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._faiss import FaissIndex, FaissNeighbors

    __all__ += ["FaissIndex", "FaissNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._hnsw import HNSWNeighbors

    __all__ += ["HNSWNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._jax import JaxNeighbors

    __all__ += ["JaxNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._numpy import NumpyNeighbors

    __all__ += ["NumpyNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._scann import ScannNeighbors

    __all__ += ["ScannNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._scipy import ScipyNeighbors

    __all__ += ["ScipyNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._sklearn import SklearnNeighbors

    __all__ += ["SklearnNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._torch import TorchNeighbors

    __all__ += ["TorchNeighbors"]
except ImportError:  # pragma: no cover
    ...

try:
    from ._usearch import UsearchIndex, UsearchNeighbors

    __all__ += ["UsearchIndex", "UsearchNeighbors"]
except ImportError:  # pragma: no cover
    ...
