import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from afp.related import dbscan, describe_pairs, shared_track_pairs


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_dbscan_matches_scikit_learn_on_core_points_and_noise():
    rng = np.random.default_rng(0)
    centres = _unit(rng.normal(size=(5, 16)))
    x = _unit(
        np.concatenate([c + rng.normal(scale=0.08, size=(40, 16)) for c in centres] + [rng.normal(size=(60, 16))])
    )
    similarity, points = 0.9, 5
    labels, core = dbscan(x, similarity, points)
    reference = DBSCAN(eps=np.sqrt(2 - 2 * similarity), min_samples=points).fit(x)
    expected_core = np.zeros(len(x), dtype=bool)
    expected_core[reference.core_sample_indices_] = True
    np.testing.assert_array_equal(core, expected_core)
    np.testing.assert_array_equal(labels == -1, reference.labels_ == -1)
    assert adjusted_rand_score(labels[core], reference.labels_[core]) == 1.0
    assert labels.max() + 1 == 5


def test_shared_track_pairs_links_tracks_with_a_common_window():
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=(6, 8)))
    shared = _unit(base[:1] + 0.01 * rng.normal(size=(1, 8)))
    vectors = np.vstack([base[:3], shared, base[3:]])
    track_id = np.array([10, 10, 10, 20, 30, 30, 30])
    labels, _ = dbscan(vectors, 0.95, 2)
    pairs = shared_track_pairs(vectors, track_id, labels)
    assert list(zip(pairs["track_a"], pairs["track_b"])) == [(10, 20)]
    assert pairs.loc[0, "max_similarity"] > 0.95

    manifest = pd.DataFrame(
        {
            "track_id": [10, 20, 30],
            "artist": ["A", "a", "B"],
            "title": ["x", "x (live)", "y"],
            "genre_top": ["Rock"] * 3,
        }
    )
    described = describe_pairs(pairs, manifest)
    assert described.loc[0, "same_artist"] and described.loc[0, "title_b"] == "x (live)"
