"""Optional CPU scorer for the candidate windows produced by LSH."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath={"contract", "reassoc"})
def best_candidate(references: np.ndarray, rows: np.ndarray, query: np.ndarray) -> tuple[int, float]:
    """Return the best row and cosine score by reading unit-vector candidates in place.

    Args:
        references: Contiguous float32 matrix of unit reference vectors.
        rows: Candidate row IDs in ascending order, so exact ties retain the first row.
        query: One unit float32 fingerprint with the same dimension as the references.

    Returns:
        The winning global row and score, or (-1, -inf) for an empty candidate list.
        Vectorised float32 accumulation can differ from BLAS by round-off.
    """
    best_row = -1
    best_score = np.float32(-np.inf)
    for row in rows:
        score = np.float32(0)
        for column in range(references.shape[1]):
            score += references[row, column] * query[column]
        if score > best_score:
            best_row = row
            best_score = score
    return best_row, best_score
