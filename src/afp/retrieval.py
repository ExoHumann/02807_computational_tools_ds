"""Exact retrieval on unit-normalised audio fingerprints."""

import numpy as np


def search(
    references: np.ndarray,
    reference_track: np.ndarray,
    queries: np.ndarray,
    threshold: float | None = None,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest-neighbour search: the track of the most similar reference window for each query.

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        reference_track: Track id of each reference fingerprint.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        threshold: Minimum cosine similarity to accept a match. Queries below it get track ``-1`` ("not found").
        chunk: Queries searched at a time, to bound memory.

    Returns:
        A tuple ``(track, score)``: the predicted track id (``-1`` if rejected) and the best cosine similarity.
    """
    best = np.concatenate(
        [(queries[i : i + chunk] @ references.T).argmax(axis=1) for i in range(0, len(queries), chunk)]
    )
    score = np.einsum("ij,ij->i", queries, references[best])
    track = reference_track[best].copy()
    if threshold is not None:
        track[score < threshold] = -1
    return track, score
