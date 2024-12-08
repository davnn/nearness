import pytest
from faiss.contrib.datasets import SyntheticDataset

from nearness import *
from .utilities import pytest_param_if_value_available

data = SyntheticDataset(
    d=128,  # data dimensionality
    nt=10_000,  # number of training points
    nb=100,  # batch query size
    nq=1,  # number of single queries
)

# we only benchmark brute-force algorithms for now
candidates = {
    "faiss": lambda: FaissNeighbors(index="Flat"),
    "hnsw": lambda: HNSWNeighbors(metric="l2", n_threads=-1, use_bruteforce=True),
    "jax": lambda: JaxNeighbors(compute_mode="use_mm_for_euclid_dist", approximate_recall_target=0.9),
    "numpy": lambda: NumpyNeighbors(metric="minkowski", p=2, compute_mode="use_mm_for_euclid_dist"),
    "scann": lambda: ScannNeighbors(search_parallel=True, use_bruteforce=True, use_tree=False, use_reorder=False),
    "scipy": lambda: ScipyNeighbors(metric="euclidean"),
    "sklearn": lambda: SklearnNeighbors(metric="euclidean", n_jobs=-1),
    "torch": lambda: TorchNeighbors(metric="minkowski", p=2, compute_mode="use_mm_for_euclid_dist"),
    "usearch": lambda: UsearchNeighbors(exact_search=True),
}
candidates = [pytest_param_if_value_available(k, v) for k, v in candidates.items()]


@pytest.mark.parametrize("n_neighbors", [1])
@pytest.mark.parametrize("model", candidates)
def test_benchmark_single_query(n_neighbors, model, benchmark):
    fit_data = data.get_train()
    query_data = data.get_queries()
    model.fit(fit_data)
    benchmark(lambda query: model.query(query, n_neighbors=n_neighbors), *query_data)


@pytest.mark.parametrize("n_neighbors", [1])
@pytest.mark.parametrize("model", candidates)
def test_benchmark_batch_query(n_neighbors, model, benchmark):
    fit_data = data.get_train()
    query_data = data.get_database()
    model.fit(fit_data)
    benchmark(lambda query: model.query_batch(query, n_neighbors=n_neighbors), query_data)
