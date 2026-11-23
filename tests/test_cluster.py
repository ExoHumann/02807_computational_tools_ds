import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score

from afp.cluster import IVFIndex, davies_bouldin, evaluate_ivf, exhaustive_search, kmeans


def _blobs(n_per: int = 100, k: int = 4, dim: int = 8, spread: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 5
    labels = np.repeat(np.arange(k), n_per)
    return centres[labels] + rng.normal(scale=spread, size=(len(labels), dim)), labels


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_kmeans_matches_scikit_learn_from_the_same_start():
    x, _ = _blobs(spread=2.0)
    start = x[np.random.default_rng(1).choice(len(x), size=5, replace=False)]
    ours = kmeans(x, 5, max_iter=300, tol=0.0, init=start)
    theirs = KMeans(5, init=start, n_init=1, max_iter=300, tol=0.0, algorithm="lloyd").fit(x)
    np.testing.assert_array_equal(ours.labels, theirs.labels_)
    np.testing.assert_allclose(ours.centroids, theirs.cluster_centers_, atol=1e-8)
    np.testing.assert_allclose(ours.inertia, theirs.inertia_, rtol=1e-8)


def test_kmeans_plus_plus_recovers_separated_blobs():
    x, truth = _blobs()
    result = kmeans(x, 4, seed=3)
    assert adjusted_rand_score(truth, result.labels) == 1.0


def test_spherical_kmeans_keeps_unit_centroids_and_handles_duplicate_starts():
    x = _unit(_blobs()[0])
    result = kmeans(x, 4, init=np.repeat(x[:1], 4, axis=0), spherical=True)
    assert np.isfinite(result.centroids).all()
    np.testing.assert_allclose(np.linalg.norm(result.centroids, axis=1), 1.0, atol=1e-6)
    assert len(np.unique(result.labels)) == 4


def test_davies_bouldin_matches_scikit_learn():
    x, truth = _blobs(spread=1.5)
    labels = kmeans(x, 6, seed=0).labels
    assert np.isclose(davies_bouldin(x, labels), davies_bouldin_score(x, labels))
    assert np.isclose(davies_bouldin(x, truth + 10), davies_bouldin_score(x, truth))
    assert davies_bouldin(x, np.zeros(len(x))) == 0.0


def test_ivf_search_is_exact_when_every_cluster_is_probed():
    rng = np.random.default_rng(0)
    references = _unit(rng.normal(size=(500, 16)))
    queries = _unit(references[:50] + rng.normal(scale=0.1, size=(50, 16)))
    index = IVFIndex.build(references, 8, sample=300, max_iter=20)
    assert index.cluster_sizes().sum() == len(references)
    exact = exhaustive_search(references, queries)
    nearest, candidates = index.search(queries, nprobe=8)
    np.testing.assert_array_equal(nearest, exact)
    assert (candidates == len(references)).all()
    _, few = index.search(queries, nprobe=1)
    assert few.mean() < len(references) / 2


def test_evaluate_ivf_tables():
    rng = np.random.default_rng(0)
    references = _unit(rng.normal(size=(400, 16)))
    track = np.repeat(np.arange(100), 4)
    queries = _unit(references[::8] + rng.normal(scale=0.05, size=(50, 16)))
    clusters, search = evaluate_ivf(references, track, queries, track[::8], ks=[4, 16], nprobes=[1, 4, 8])
    assert list(clusters["k"]) == [4, 16]
    assert len(search) == 1 + 2 + 3
    exact = search[search["k"] == 0].iloc[0]
    assert exact["top1"] == 1.0
    assert search["top1"].between(0, 1).all() and search["agree"].between(0, 1).all()
    full = search[(search["k"] == 4) & (search["nprobe"] == 4)].iloc[0]
    assert full["agree"] == 1.0
