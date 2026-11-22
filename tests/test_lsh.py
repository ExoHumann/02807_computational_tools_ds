"""Small deterministic checks for the release-fingerprint retrieval layer."""

import numpy as np
import pytest

from afp.lsh import AngularLSH
from afp.retrieval import search


def test_duplicate_windows_tie_break_and_candidate_union():
    """Multiple tables must not duplicate a row, and ties go to the first reference."""
    references = np.array([[1, 0], [1, 0]], dtype=np.float32)
    tracks = np.array([12, 42])
    index = AngularLSH(references, tracks, bits=4, tables=8)
    matches = index.search(references)
    expected_tracks, expected_scores = search(references, tracks, references)
    np.testing.assert_array_equal(matches.track, expected_tracks)
    np.testing.assert_allclose(matches.score, expected_scores)
    np.testing.assert_array_equal(matches.candidates, [2, 2])


def test_empty_buckets_and_cosine_rejection():
    """An empty union and a nonempty union below the threshold both reject the query."""
    references = np.array([[1, 0]], dtype=np.float32)
    index = AngularLSH(references, np.array([8]), bits=2, tables=8)
    queries = np.array([[-1, 0], [0.999, np.sqrt(1 - 0.999**2)]], dtype=np.float32)
    result = index.search(queries, threshold=1.0)
    np.testing.assert_array_equal(result.track, [-1, -1])
    np.testing.assert_array_equal(result.candidates, [0, 1])
    np.testing.assert_allclose(result.score, [-np.inf, 0.999])


def test_query_margin_probes_only_add_candidates():
    """Probing the least-certain table bits keeps the base buckets and correct reranking."""
    rng = np.random.default_rng(7)
    references = rng.normal(size=(32, 4)).astype(np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    queries = references[:5] + rng.normal(scale=0.1, size=(5, 4)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    tracks = np.arange(len(references))
    index = AngularLSH(references, tracks, bits=4, tables=8, seed=4)

    base = index.search(queries)
    probed = index.search(queries, probe_budget=4)

    assert np.all(probed.candidates >= base.candidates)
    assert np.all(probed.score >= base.score)
    assert np.all(probed.track == tracks[:5])
    np.testing.assert_array_equal(index.search(queries).track, base.track)


def test_query_margin_probe_budget_validation_and_empty_batch():
    """The budget is bounded by table count and empty batches preserve their dimensions."""
    index = AngularLSH(np.array([[1, 0]], dtype=np.float32), np.array([8]), bits=2, tables=3)
    result = index.search(np.empty((0, 2), dtype=np.float32), probe_budget=3)
    np.testing.assert_array_equal(result.track, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(result.candidates, np.empty(0, dtype=np.int64))

    try:
        index.search(np.array([[1, 0]], dtype=np.float32), probe_budget=4)
    except ValueError as error:
        assert "number of tables" in str(error)
    else:
        raise AssertionError("A probe budget larger than the table count should fail")


def test_numba_candidate_reranking_matches_numpy():
    """The optional in-place scorer keeps the same track IDs and cosine scores."""
    pytest.importorskip("numba")
    rng = np.random.default_rng(70)
    references = rng.normal(size=(40, 12)).astype(np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    queries = references[:5] + rng.normal(scale=0.2, size=(5, 12)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    index = AngularLSH(references, np.arange(len(references)), bits=4, tables=12, seed=22)

    numpy_matches = index.search(queries, probe_budget=4)
    numba_matches = index.search(queries, probe_budget=4, rerank_backend="numba")

    np.testing.assert_array_equal(numba_matches.track, numpy_matches.track)
    np.testing.assert_allclose(numba_matches.score, numpy_matches.score, atol=2e-7)
    np.testing.assert_array_equal(numba_matches.candidates, numpy_matches.candidates)
