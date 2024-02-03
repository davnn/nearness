from uuid import uuid4
import threading
from dataclasses import dataclass

import faiss
from faiss.contrib.datasets import SyntheticDataset
from typing_extensions import Self, Literal

from numpy.typing import DTypeLike
import hypothesis
import hypothesis.strategies as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.datasets import load_digits

import nearness
from nearness import *
from nearness._base import InvalidSignatureError
from .utilities import *

hypothesis.settings.register_profile("no-deadline", deadline=None)
hypothesis.settings.load_profile("no-deadline")


@dataclass
class Candidate:
    """Create a test candidate to be checked against a corresponding scikit-learn reference implementation."""

    implementation: NearestNeighbors
    reference: SklearnNeighbors | None = None


@dataclass
class Data:
    fit: np.ndarray
    query: np.ndarray
    batch: np.ndarray


candidates = {
    # comparing two equal models might seem unnecessary, but some tests are only performed on the ``implementation``
    "sklearn": lambda: Candidate(
        implementation=SklearnNeighbors(metric="minkowski", p=2), reference=SklearnNeighbors(metric="minkowski", p=2)
    ),
    "numpy": lambda: Candidate(
        implementation=NumpyNeighbors(metric="minkowski", p=2, compute_mode="donot_use_mm_for_euclid_dist"),
        reference=SklearnNeighbors(metric="minkowski", p=2),
    ),
    "torch": lambda: Candidate(
        implementation=TorchNeighbors(metric="minkowski", p=2, compute_mode="donot_use_mm_for_euclid_dist"),
        reference=SklearnNeighbors(metric="minkowski", p=2),
    ),
    # it's important to only use a single tree for testing (with little data), otherwise it's possible that
    # some points are never visited
    "annoy": lambda: Candidate(
        implementation=AnnoyNeighbors(metric="euclidean", n_trees=1, n_search_neighbors=32, random_seed=0),
        reference=SklearnNeighbors(metric="minkowski", p=2),
    ),
    "hnsw-brute": lambda: Candidate(
        implementation=HNSWNeighbors(metric="l2", use_bruteforce=True), reference=SklearnNeighbors(metric="sqeuclidean")
    ),
    "hnsw": lambda: Candidate(
        implementation=HNSWNeighbors(metric="l2", n_index_neighbors=32, n_search_neighbors=32, n_links=32),
        reference=SklearnNeighbors(metric="sqeuclidean"),
    ),
    "scipy": lambda: Candidate(
        implementation=ScipyNeighbors(metric="euclidean"), reference=SklearnNeighbors(metric="euclidean")
    ),
    "autofaiss": lambda: Candidate(
        implementation=AutoFaissNeighbors(metric_type="l2"), reference=SklearnNeighbors(metric="sqeuclidean")
    ),
    "faiss": lambda: Candidate(
        implementation=FaissNeighbors(index_or_factory="Flat"), reference=SklearnNeighbors(metric="sqeuclidean")
    ),
    "scann": lambda: Candidate(
        implementation=ScannNeighbors(use_bruteforce=True, use_tree=False, use_reorder=False),
        reference=SklearnNeighbors(metric="sqeuclidean"),
    ),
    "jax": lambda: Candidate(
        implementation=JaxNeighbors(compute_mode="donot_use_mm_for_euclid_dist"),
        reference=SklearnNeighbors(metric="euclidean"),
    ),
}

candidates = [pytest_param_if_value_available(k, v) for k, v in candidates.items()]
DataGenerationMethod = Literal["random", "mds", "digits", "synthetic"]


def make_data(
    dim: int,
    fit_size: int = 1,
    batch_size: int = 1,
    dtype: DTypeLike = np.float64,
    method: DataGenerationMethod = "random",
) -> Data:
    # TODO: Evaluate if it makes sense to add more realistic datasets for testing and benchmarking, such as:
    # https://github.com/facebookresearch/faiss/blob/main/contrib/datasets.py
    rs = np.random.RandomState(0)

    if method == "digits":
        x, _ = load_digits(return_X_y=True)
        x = PCA(n_components=dim, random_state=rs).fit_transform(x).astype(dtype)
        fit, batch = train_test_split(x, train_size=fit_size, test_size=batch_size, random_state=rs)
        return Data(fit=fit, batch=batch, query=batch[0])

    if method == "mds":
        mds = MDS(
            n_components=dim,
            dissimilarity="precomputed",
            normalized_stress="auto",
            n_init=1,
            max_iter=10,
            random_state=rs,
        )
        n_data = fit_size + batch_size + 1
        x = np.arange(start=1, stop=n_data * n_data + 1)
        rs.shuffle(x)
        x = x.reshape(n_data, n_data)
        x = np.tril(x, k=-1)
        x = x + x.T
        x = mds.fit_transform(x).astype(dtype)
        batch = x[:batch_size]
        fit = x[batch_size:]
        return Data(fit=fit, batch=batch, query=batch[0])

    if method == "random":
        fit = rs.rand(fit_size, dim).astype(dtype)
        batch = rs.rand(batch_size, dim).astype(dtype)
        query = rs.rand(dim).astype(dtype)
        return Data(fit=fit, batch=batch, query=query)

    if method == "synthetic":
        data = SyntheticDataset(d=dim, nt=fit_size, nb=batch_size, nq=1)
        return Data(fit=data.get_train(), batch=data.get_database(), query=data.get_queries()[0])

    msg = f"Invalid data generation method, found {method}."
    raise ValueError(msg)


data_strategy = st.builds(
    make_data,
    dim=st.integers(min_value=1, max_value=32),
    fit_size=st.integers(min_value=1, max_value=32),
    batch_size=st.integers(min_value=1, max_value=32),
    dtype=st.sampled_from([np.float32, np.float64]),
)


@st.composite
def neighbors_strategy(draw):
    data = draw(data_strategy)
    max_neighbors = len(data.fit)
    n_neighbors = st.integers(min_value=1, max_value=max_neighbors)
    return data, draw(n_neighbors)


@pytest.mark.parametrize("method", ["query", "query_idx", "query_dist"])
@pytest.mark.parametrize("candidate", candidates)
def test_query_fail_not_fit(method, candidate):
    # This test should run before any .fit has been called on ``implementation``.
    vec = np.zeros(1, dtype=np.float64)
    implementation = candidate.implementation

    with pytest.raises(Exception):
        getattr(implementation, method)(vec, n_neighbors=1)


@pytest.mark.parametrize("method", ["query_batch", "query_batch_idx", "query_batch_dist"])
@pytest.mark.parametrize("candidate", candidates)
def test_query_batch_fail_not_fit(method, candidate):
    # This test should run before any .fit has been called on ``implementation``.
    mat = np.zeros((1, 1), dtype=np.float64)
    implementation = candidate.implementation

    with pytest.raises(Exception):
        getattr(implementation, method)(mat, n_neighbors=1)


@hypothesis.given(data=data_strategy)
@pytest.mark.parametrize("candidate", candidates)
def test_fit_return_self(data, candidate):
    # Test if the fit method works correctly
    model = candidate.implementation
    assert model == model.fit(data.fit)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@pytest.mark.parametrize("candidate", candidates)
def test_query(data_and_neighbors, candidate):
    if candidate.reference is None:
        pytest.skip()

    data, n_neighbors = data_and_neighbors

    candidate.reference.fit(data.fit)
    candidate.implementation.fit(data.fit)

    idx_ref, dist_ref = candidate.reference.query(data.query, n_neighbors=n_neighbors)
    idx_imp, dist_imp = candidate.implementation.query(data.query, n_neighbors=n_neighbors)

    assert array_equal(idx_ref, idx_imp)
    assert approx_equal(dist_ref, dist_imp)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@pytest.mark.parametrize("candidate", candidates)
def test_query_idx_dist(data_and_neighbors, candidate):
    data, n_neighbors = data_and_neighbors

    candidate.implementation.fit(data.fit)
    idx_ref, dist_ref = candidate.implementation.query(data.query, n_neighbors=n_neighbors)
    idx_check = candidate.implementation.query_idx(data.query, n_neighbors=n_neighbors)
    dist_check = candidate.implementation.query_dist(data.query, n_neighbors=n_neighbors)

    assert array_equal(idx_ref, idx_check)
    assert approx_equal(dist_ref, dist_check)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@pytest.mark.parametrize("candidate", candidates)
def test_query_batch(data_and_neighbors, candidate):
    if candidate.reference is None:
        pytest.skip()

    data, n_neighbors = data_and_neighbors

    candidate.reference.fit(data.fit)
    candidate.implementation.fit(data.fit)

    idx_ref, dist_ref = candidate.reference.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_imp, dist_imp = candidate.implementation.query_batch(data.batch, n_neighbors=n_neighbors)

    assert array_equal(idx_ref, idx_imp)
    assert approx_equal(dist_ref, dist_imp)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@pytest.mark.parametrize("candidate", candidates)
def test_query_batch_idx_dist(data_and_neighbors, candidate):
    data, n_neighbors = data_and_neighbors

    candidate.implementation.fit(data.fit)
    idx_ref, dist_ref = candidate.implementation.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_check = candidate.implementation.query_batch_idx(data.batch, n_neighbors=n_neighbors)
    dist_check = candidate.implementation.query_batch_dist(data.batch, n_neighbors=n_neighbors)

    assert array_equal(idx_ref, idx_check)
    assert approx_equal(dist_ref, dist_check)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
@pytest.mark.parametrize("candidate", candidates)
def test_save_load_identity(data_and_neighbors, candidate, tmp_path, request):
    key = request.node.callspec.id
    save_path = tmp_path / uuid4().hex

    if key == "hnsw-brute":
        pytest.skip("Saving of HSNW index with use_bruteforce=True is not possible currently.")

    data, n_neighbors = data_and_neighbors
    saved_model = candidate.implementation.fit(data.fit)
    saved_model.save(save_path)
    loaded_model = candidate.implementation.load(save_path)

    idx_save, dist_save = saved_model.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_load, dist_load = loaded_model.query_batch(data.batch, n_neighbors=n_neighbors)

    assert array_equal(idx_save, idx_load)
    assert approx_equal(dist_save, dist_load)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_annoy_save_load_identity(data_and_neighbors, tmp_path):
    """Must use unique model names for both models, because only one ``tmp_path`` is created (function-scoped)."""
    data, n_neighbors = data_and_neighbors
    _, n_dim = data.fit.shape

    # build models and save
    run_id = uuid4().hex
    save_path, disk_path = tmp_path / f"{run_id}-save.annoy", tmp_path / f"{run_id}-disk.annoy"

    args = dict(n_trees=1, n_search_neighbors=32, load_index_dim=n_dim)
    save_model = AnnoyNeighbors(**args, save_index_path=save_path)
    save_model.fit(data.fit)
    disk_model = AnnoyNeighbors(**args, disk_build_path=disk_path)
    disk_model.fit(data.fit)

    # load models
    load_model_save = AnnoyNeighbors(**args, load_index_path=save_path)
    load_model_disk = AnnoyNeighbors(**args, load_index_path=disk_path)

    idx_save, dist_save = save_model.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_disk, dist_disk = disk_model.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_load_save, dist_load_save = load_model_save.query_batch(data.batch, n_neighbors=n_neighbors)
    idx_load_disk, dist_load_disk = load_model_disk.query_batch(data.batch, n_neighbors=n_neighbors)

    # save models equal load models
    assert array_equal(idx_save, idx_disk, idx_load_save, idx_load_disk)
    assert approx_equal(dist_save, dist_disk, dist_load_save, dist_load_disk)


@hypothesis.given(data_and_neighbors=neighbors_strategy())
def test_faiss_index_creation(data_and_neighbors):
    data, n_neighbors = data_and_neighbors
    model_str = FaissNeighbors(index_or_factory="Flat")
    model_str.fit(data.fit)
    model_fun = FaissNeighbors(index_or_factory=lambda d: faiss.IndexFlat(d))
    model_fun.fit(data.fit)
    model_raw = FaissNeighbors(index_or_factory=faiss.IndexFlat)
    model_raw.fit(data.fit)

    idx_str, dist_str = model_str.query(data.query, n_neighbors)
    idx_fun, dist_fun = model_fun.query(data.query, n_neighbors)
    idx_raw, dist_raw = model_raw.query(data.query, n_neighbors)

    assert array_equal(idx_str, idx_fun, idx_raw)
    assert approx_equal(dist_str, dist_fun, dist_raw)


def test_thread_safety():
    # Number of threads to create
    num_threads = 100
    original_thread_ids = set()
    captured_thread_ids = set()

    class Model(NearestNeighbors):
        def __init__(self, *, thread_id: int):
            super().__init__()

        def fit(self, data: np.ndarray) -> Self: ...

        def query(self, point: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]: ...

    # Use a threading.Event to synchronize threads
    start_event = threading.Event()

    def create_instance():
        # Wait for the start signal
        start_event.wait()

        # Get the current thread id
        thread_id = threading.current_thread().ident
        original_thread_ids.add(thread_id)

        # Set the thread id on the model
        model = Model(thread_id=thread_id)
        captured_id = model.parameters.thread_id
        captured_thread_ids.add(captured_id)

        # Perform any assertions on the instance
        assert thread_id == model.parameters.thread_id

    # Create and start threads
    threads = [threading.Thread(target=create_instance) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    # Signal threads to start creating instances
    start_event.set()

    # Join threads
    for thread in threads:
        thread.join()

    # Perform any additional assertions or checks
    assert len(captured_thread_ids) == len(original_thread_ids)
    assert captured_thread_ids == original_thread_ids


def test_missing_abstract_method():
    class N(NearestNeighbors): ...

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        N()


def test_keyword_only():
    with pytest.raises(InvalidSignatureError):

        class N(NearestNeighbors):
            def __init__(self, a): ...


def test_warn_check():
    class ModelNoConfig(NearestNeighbors):
        no_method = True

        def __init__(self): ...

        def fit(self, data): ...

        def query(self, point, n_neighbors): ...

    class ModelWrongAttribute(NearestNeighbors):
        no_method = True

        def __init__(self):
            self.config.methods_require_fit = self.config.methods_require_fit | {"no_method"}

        def fit(self, data): ...

        def query(self, point, n_neighbors): ...

    class ModelMissingAttribute(NearestNeighbors):
        def __init__(self):
            self.config.methods_require_fit = self.config.methods_require_fit | {"missing_method"}

        def fit(self, data): ...

        def query(self, point, n_neighbors): ...

    with pytest.warns(UserWarning, match="Attempting to enable '__fitted__' check for invalid attribute"):
        ModelWrongAttribute()

    with pytest.warns(UserWarning, match="Attempting to enable '__fitted__' check for missing attribute"):
        ModelMissingAttribute()

    with pytest.warns(UserWarning, match="Attempting to enable '__fitted__' check for invalid attribute"):
        model = ModelNoConfig()
        model.config.methods_require_fit = model.config.methods_require_fit | {"no_method"}

    with pytest.warns(UserWarning, match="Attempting to enable '__fitted__' check for missing attribute"):
        model = ModelNoConfig()
        model.config.methods_require_fit = model.config.methods_require_fit | {"missing_method"}


def test_check_attribute():
    class Model(NearestNeighbors):
        def __init__(self): ...

        def fit(self, data):
            return self

        def query(self, point, n_neighbors): ...

    model = Model()

    # make sure that all methods have a ``__check__`` attribute
    for method in model.config.methods_require_fit:
        assert hasattr(getattr(model, method), "__check__")

    # removing methods should also remove their check
    model.config.methods_require_fit = model.config.methods_require_fit - {"query"}
    assert not hasattr(model.query, "__check__")

    # adding methods should also add their check
    model.config.methods_require_fit = model.config.methods_require_fit | {"query"}
    assert hasattr(model.query, "__check__")

    # make sure that the checks are removed after fitting
    model.fit(None)

    # now after ``fit``, the check should be removed from all methods
    for method in model.config.methods_require_fit:
        assert not hasattr(getattr(model, method), "__check__")


def test_retrieve_version():
    nearness.get_version()
