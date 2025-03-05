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
neighbors search implementation. The following implementations are available:

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
- [Usearch](https://github.com/unum-cloud/usearch) exposes ``UsearchNeighbors``

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
    """Given a vector ``point``, search its ``n_neighbors``, returning the indices and distances."""
    ...


def query_batch(points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Given a matrix ``points``, search their ``n_neighbors`` returning the indices and distances."""
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
``query`` methods. By default, ``query_batch`` uses [joblib](https://github.com/joblib/joblib) to process a batch of
queries in a threadpool, but most of the time you'd want to implement ``query_batch`` on your own for improved
efficiency.

The following example illustrates the concepts of ``config`` and ``parameters``.

```python
import numpy as np
from nearness import NearestNeighbors


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

### Available Algorithms

### `AnnoyNeighbors`

This class implements the nearest neighbors using the **Annoy** algorithm.

#### Parameters:

- **metric** (`str`): One of `["angular", "euclidean", "manhattan", "hamming", "dot"]`.
- **n_trees** (`int`): Builds a forest of `n_trees` trees. More trees give higher precision when querying.
- **n_search_neighbors** (`int | None`): Inspect up to `n_search_neighbors` nodes, default is `n_search_neighbors` * n.
- **random_seed** (`int | None`): Initialize the random number generator with the given seed.
- **disk_build_path** (`str | Path | None`): Build the index on disk at the given path.
- **save_index_path** (`str | Path | None`): Save the index to disk at the given path.
- **load_index_path** (`str | Path | None`): Loads (mmaps) an index from disk from the given path.
- **load_index_dim** (`int | None`): Specify the dimension for a loaded index.
- **prefault** (`bool`): If set to `True`, it will pre-read the entire file into memory (mmap MAP_POPULATE).

---

### `AutoFaissNeighbors`

This class implements the nearest neighbors using **AutoFaiss**.

#### Parameters:

- **save_on_disk** (`bool`): Whether to save the index on disk, default is `False`.
- **pre_load_index** (`bool`): Pre-load the index directly on `__init__`.
- **pre_load_using_mmap** (`bool`): Use mmap to pre-load the index on `__init__`.
- **index_path** (`str | None`): Destination path of the quantized model.
- **index_infos_path** (`str | None`): Destination path of the metadata file.
- **ids_path** (`str | None`): Path where the mapping files Ids->vector index will be stored in parquet format (only for
  `parquet`).
- **file_format** (`str`): File format, either `"npy"` or `"parquet"`; default is `"npy"`.
- **embedding_column_name** (`str`): Embeddings column name for parquet; default is `"embedding"`.
- **id_columns** (`list[str] | None`): Column names containing vector IDs (only for `parquet`); generates mapping files;
  default is `None`.
- **index_key** (`str | None`): String for index factory; if `None`, an index is chosen heuristically.
- **index_param** (`str | None`): Hyperparameters for the index; if `None`, chosen heuristically.
- **max_index_query_time_ms** (`int | float`): Approximate bound on query time for KNN search.
- **max_index_memory_usage** (`str`): Maximum allowed size for the index (strict limit).
- **min_nearest_neighbors_to_retrieve** (`int`): Minimum number of nearest neighbors to retrieve; overrides query time
  limit.
- **current_memory_available** (`str`): Available memory for index creation; more memory improves performance.
- **use_gpu** (`bool`): Experimental; enables GPU training but is untested.
- **metric_type** (`str`): Similarity function for queries: `"ip"` (inner product) or `"l2"` (Euclidean distance).
- **nb_cores** (`int | None`): Number of cores to use; attempts to guess if not provided.
- **make_direct_map** (`bool`): Creates a direct map for embeddings reconstruction (IVF indices only); increases RAM
  usage.
- **should_be_memory_mappable** (`bool`): If `True`, selects only indices that can be memory-mapped on disk; default is
  `False`.
- **distributed** (`str | None`): If `"pyspark"`, builds indices using PySpark (only supports `parquet`).
- **temporary_indices_folder** (`str`): Folder for temporary indices generated by each Spark executor (only for
  `"pyspark"`).
- **verbose** (`int`): Verbosity level, set via logging; default is `logging.INFO`.
- **nb_indices_to_keep** (`int`): Max indices to keep when distributed = `"pyspark"`; enables building larger indices.

---

### `FaissNeighbors`

This class implements nearest neighbors using **Faiss**.

#### Parameters:

- **index** (`str | FaissIndex | faiss.Index`): An index factory string, a FaissIndex wrapped index or a faiss.Index.
- **add_data_on_fit** (`bool`): Add the data used for index training to the learned index.

---

### `HNSWNeighbors`

This class implements nearest neighbors using **HNSW** (Hierarchical Navigable Small World graphs).

#### Parameters:

- **metric** (`str`): One of `["l2", "ip", "cosine"]`.
- **n_index_neighbors** (`int`): Size of the dynamic neighbors candidate list during index construction.
- **n_search_neighbors** (`int | None`): Size of the dynamic neighbors candidate list during index search.
- **n_links** (`int`): Number of connections per node in the graph. Higher values improve accuracy but use more memory.
- **n_threads** (`int`): Number of threads to use during index search.
- **random_seed** (`int`): Seed for random number generation, ensuring reproducibility across runs.
- **use_bruteforce** (`bool`): Skip index creation and use brute-force search over all items instead.

---

### `JaxNeighbors`

This class implements nearest neighbors using **Jax**.

#### Parameters:

- **metric** (`str`): Only `"minkowski"` is supported currently.
- **p** (`int`): Parameter that defines the specific p-norm used.
- **compute_mode** (`str`): Use matrix multiplication when `p=2` and mode is `"use_mm_for_euclid_dist"`.
- **approximate_recall_target** (`float`): Recall target for nearest neighbors sorting.

---

### `NumpyNeighbors`

This class implements nearest neighbors using **Numpy**.

#### Parameters:

- **metric** (`str`): Only `"minkowski"` is supported currently.
- **p** (`int`): Parameter that defines the specific p-norm used.
- **compute_mode** (`str`): Use matrix multiplication when `p=2` and mode is `"use_mm_for_euclid_dist"`.

---

### `ScannNeighbors`

This class implements nearest neighbors using **Scann**.

#### Parameters:

- **metric** (`str`): One of `["dot_product", "squared_l2"]`.
- **n_neighbors** (`int`): Number of neighbors specified for index creation (overridden on query).
- **n_training_threads** (`int`): Number of threads used for index creation.
- **use_tree** (`bool`): Use tree for data partitioning.
- **use_bruteforce** (`bool`): Use brute-force approach for scoring, otherwise use asymmetric hashing (AH).
- **use_reorder** (`bool`): Use rescoring of results, highly recommended if AH scoring is used.
- **search_parallel** (`bool`): Perform batched searches in parallel.
- **tree_config** (`ScannTreeConfig | None`): Configuration parameters for tree partitioning.
- **bruteforce_config** (`ScannBruteForceConfig | None`): Configuration parameters for brute-force search.
- **hashing_config** (`ScannHashingConfig | None`): Configuration parameters for asymmetric hashing search.
- **reorder_config** (`ScannReorderConfig | None`): Configuration parameters for score reordering (rescoring).
- **search_config** (`ScannSearchConfig | None`): Configuration parameters for searches.

---

### `ScipyNeighbors`

This class implements exact nearest neighbors using **SciPy**.

#### Parameters:

- **metric** (`str`): One of the metrics available in `scipy.cdist`.
- **metric_args** (`dict` | `None`): Dictionary of parameters used for the chosen metric.

---

### `SklearnNeighbors`

This class implements nearest neighbors using **Scikit-learn**.

#### Parameters:

- **algorithm** (`str`): One of `["auto", "brute", "ball_tree", "kd_tree"]`.
- **leaf_size** (`int`): Leaf size passed to `BallTree` or `KDTree`.
- **metric** (`str`): One of the metrics listed in `sklearn.metrics.pairwise.distance_metrics()`.
- **p** (`int`): Parameter for the Minkowski metric that defines the specific p-norm used.
- **metric_params** (`dict[str, Any] | None`): Additional keyword arguments for the metric function.
- **n_jobs** (`int | None`): The number of parallel jobs to run for neighbors search.

---

### `TorchNeighbors`

This class implements nearest neighbors using **Torch**.

#### Parameters:

- **metric** (`str`): Only `"minkowski"` is supported currently.
- **p** (`int`): Parameter that defines the specific p-norm used.
- **compute_mode** (`str`): Use matrix multiplication when `p=2` and mode is `"use_mm_for_euclid_dist"`.
- **force_dtype** (`torch.dtype | None`): Ensure a specific dtype is used for search.
- **force_device** (`torch.device | str | None`): Ensure a specific device is used for search.

---

### `UsearchNeighbors`

This class implements nearest neighbors using **Usearch**.

#### Parameters:

- **index** (`usearch.Index | UsearchIndex`): A `UsearchIndex` wrapped index or a `usearch.Index`.
- **exact_search** (`bool`): Bypass index and use brute-force exact search.
- **copy_data** (`bool`): Should the index store a copy of vectors.
- **threads_fit** (`int`): Optimal number of cores to use for index creation.
- **threads_search** (`int`): Optimal number of cores to use for index search.
- **log_fit** (`bool`): Whether to print the progress bar on index creation.
- **log_search** (`bool`): Whether to print the progress bar on index search.
- **progress_fit** (`usearch.ProgressCallback | None`): Callback to report stats of the index creation progress.
- **progress_search** (`usearch.ProgressCallback | None`): Callback to report stats of the index search progress.
- **progress_save** (`usearch.ProgressCallback | None`): Callback to report stats of the index save progress.
- **progress_load** (`usearch.ProgressCallback | None`): Callback to report stats of the index load progress.
- **save_index_path** (`str | Path | None`): Save the index after index creation.
- **load_index_path** (`str | Path | None`): Load an existing index directly on `__init__`.
- **map_file_index** (`bool`): Memory map an index after creation or on load.
- **add_data_on_fit** (`bool`): Add data to the index directly on index creation.
