<p align="center">
  <img src="https://raw.githubusercontent.com/davnn/nearness/main/assets/nearness.png">
</p>

-------------------------------------------------------------------------------------------

[![Check Status](https://github.com/davnn/nearness/actions/workflows/check.yml/badge.svg)](https://github.com/davnn/nearness/actions?query=workflow%3Acheck)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/davnn/nearness/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/davnn/nearness/releases)
![Coverage Report](https://raw.githubusercontent.com/davnn/nearness/main/assets/coverage.svg)

*nearness* is a unified interface for (approximate) nearest neighbors search.

Using ``pip install nearness`` only installs the interface and does not add any concrete nearest
neighbors search implementation. The following implementations are available currently:

- [Annoy](https://github.com/spotify/annoy) exposes ``AnnoyNeighbors``
- [AutoFaiss](https://github.com/criteo/autofaiss) exposes ``AutoFaissNeighbors``
- [Faiss](https://github.com/facebookresearch/faiss) exposes ``FaissNeighbors``
- [PyGlass](https://github.com/zilliztech/pyglass) exposes ``GlassNeighbors``
- [HNSWLib](https://github.com/nmslib/hnswlib) exposes ``HNSWNeighbors``
- [Jax](https://github.com/google/jax) exposes ``JaxNeighbors``
- [Numpy](https://github.com/numpy/numpy) exposes ``NumpyNeighbors``
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) exposes ``ScannNeighbors``
- [SciPy](https://github.com/scipy/scipy) exposes ``ScipyNeighbors``
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) exposes ``SklearnNeighbors``
- [PyTorch](https://github.com/pytorch/pytorch) exposes ``TorchNeighbors``

Installing one of the above packages exposes the corresponding nearest neighbors implementation. For example,
``nearness.FaissNeighbors`` is available if [Faiss](https://github.com/facebookresearch/faiss) is installed.

Another option to install the underlying packages is to specify them as package extras, e.g.
``pip install nearness[faiss]`` installs the nearness with ``faiss-cpu``. If you require flexibility regarding
the specific version of the installed packages, it's recommended to install them explicitly.

### API

The nearness API consists of a single class called ``NearestNeighbors`` with the following methods.

```python
def fit(data: np.ndarray) -> Self:
    """Learn an index structure based on a matrix of points."""
    ...


def query(point: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Search ``n_neighbors`` for a single point, returning the indices and distances."""
    ...


def query_batch(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Search ``n_neighbors`` for a batch of points, returning the indices and distances."""
    ...


def save(file: str | Path) -> None:
    """Save the state of the model using pickle such that it can be fully restored."""
    ...


def load(file: str | Path) -> None:
    """Load a model using pickle to fully restore the saved state."""
    ...
```

The interface to all methods is based on [NumPy](https://github.com/numpy/numpy) arrays, but implementations might
``overload`` the methods such that other data types are supported. For example, ``TorchNeighbors`` supports NumPy and
PyTorch arrays.

The library additionally exports a global ``config`` object, of which the current state is passed to any
``NearestNeighbors`` class instantiation. Any modifications of a class-bound config is then specific to the class
and does not modify the global object.

In addition to the global config, we treat all of the ``__init__`` arguments to ``NearestNeighbors``
as ``parameters`` of the class, automagically binding the parameters to an object before instantiation. We expose the
config and parameters of an object as ``obj.config`` and ``obj.parameters``.

### Usage Example

The following example demonstrates how to use nearness given that
[scikit-learn](https://github.com/scikit-learn/scikit-learn) is installed.

```python
from nearness import SklearnNeighbors
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, _ = load_digits(return_X_y=True)
X_train, X_test = train_test_split(X)

# create a brute force nearest neighbors model
model = SklearnNeighbors(algorithm="brute")
model.fit(X_train)

# query a single test point
idx, dist = model.query(X_test[0], n_neighbors=5)

# query all test points
idx_batch, dist_batch = model.query_batch(X_test, n_neighbors=5)

# change the algorithm to a K-D tree and fit again
model.parameters.algorithm = "kd_tree"
model.fit(X_train)

# save the model to a file
model.save("my_sklearn_model")

# load the model from file
kdtree_model = SklearnNeighbors.load("my_sklearn_model")

# query again using the loaded model
kdtree_model.query(X_test[0], n_neighbors=5)
```

### Algorithm Implementation

To define your own ``NearestNeighbors`` algorithm it is only necessary to implement above specified ``fit`` and
``query`` methods. By default, ``query_batch`` uses a joblib to process a batch of queries in a threadpool, but most of
the time you'd want to implement ``query_batch`` on your own for improved efficiency.

The following example illustrates the concepts of ``config`` and ``parameters``.

```python
class MyNearestNeighbors(NearestNeighbors):
    # only keyword-only arguments are allowed for subclasses of ``NearestNeighbors``.
    def __init__(self, *, a: int = 0):
        # the __init__ parameters are injected as ``parameters``
        print(self.parameters.a)  # 0

        # the parameters can be modified as needed
        self.parameters.a += 1
        print(self.parameters.a)  # 1

        # a copy of the current global configuration is injected as ``config``
        print(self.config.save_compression)  # 0

        # the configuration can be modified as needed (does not modify the global config)
        self.config.save_compression = 1
        print(self.config.save_compression)  # 1

    def fit(self, data: np.ndarray) -> "Self":
        ...

    def query(self, point: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        ...
```

An interesting configuration aspect is ``methods_require_fit``, which specifies the set of methods that require a
successful call of ``fit`` before they can be used. By default, the query methods are listed in
``methods_require_fit``, and, if a query method is called before ``fit``, an informative error message is shown.
A successful fit additionally sets the ``is_fitted`` property to ``True`` and removes the fit checks such that
there is zero overhead for queries. Manually setting ``is_fitted`` to ``False`` again adds the
checks to all methods specified in ``methods_require_fit``.
