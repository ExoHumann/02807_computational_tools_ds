"""Small deterministic checks for the release-fingerprint retrieval layer."""

import numpy as np

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
