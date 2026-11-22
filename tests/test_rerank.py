"""Check the optional fused scorer against the NumPy retrieval contract."""

import numpy as np
import pytest

pytest.importorskip("numba")
from afp.rerank import best_candidate


def test_fused_score_uses_global_row_ids_and_handles_ties_and_misses():
    """Ties retain the first candidate even if the global best row is outside the subset."""
    references = np.array([[1, 0], [0.8, 0.6], [0.8, 0.6], [-1, 0]], dtype=np.float32)
    row, score = best_candidate(references, np.array([1, 2, 3]), references[0])
    assert row == 1
    assert score == pytest.approx(0.8)
    assert best_candidate(references, np.array([], dtype=np.int64), references[0]) == (-1, -np.inf)


def test_fused_scores_agree_with_cosine_search_on_a_subset():
    """Random unit vectors produce the same winning row as gathering into a dense matrix."""
    rng = np.random.default_rng(19)
    references = rng.standard_normal((190, 256), dtype=np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    rows = np.sort(rng.choice(len(references), 37, replace=False))
    query = references[0]
    scores = references[rows] @ query
    row, score = best_candidate(references, rows, query)
    assert row == rows[scores.argmax()]
    assert score == pytest.approx(scores.max(), abs=2e-6)
