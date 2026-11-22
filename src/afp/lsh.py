"""Random-hyperplane LSH for unit-normalised audio fingerprints."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    """Track IDs, cosine scores and unique reference-window counts, in query order."""

    track: np.ndarray
    score: np.ndarray
    candidates: np.ndarray


class AngularLSH:
    """Index reference windows by signs of independent Gaussian projections.

    A query takes the union of its matching buckets and reranks those windows
    by cosine similarity. The smallest reference row wins exact score ties,
    matching NumPy argmax. Empty buckets return track -1 and score -inf.
    """

    def __init__(
        self, references: np.ndarray, track_ids: np.ndarray, *, bits: int, tables: int, seed: int = 2807
    ) -> None:
        """Build the index once from a non-empty matrix and its integer track IDs."""
        if references.ndim != 2 or not references.size or track_ids.shape != (len(references),):
            raise ValueError("Expected a non-empty reference matrix and one track ID per row")
        if not 1 <= bits <= 16 or tables < 1:
            raise ValueError("Use 1–16 bits per table and at least one table")
        self.references = np.ascontiguousarray(references, dtype=np.float32)
        self.track_ids = np.asarray(track_ids, dtype=np.int64)
        self.tables = tables
        self.bits = bits
        rng = np.random.default_rng(seed)
        planes = rng.standard_normal((tables, bits, references.shape[1]), dtype=np.float32)
        self.planes = np.ascontiguousarray(planes.reshape(tables * bits, -1).T)
        self.weights = 1 << np.arange(bits, dtype=np.int32)
        self.buckets = []
        # Build a table at a time to bound temporary memory on larger catalogues.
        for table in range(tables):
            cols = slice(table * bits, (table + 1) * bits)
            keys = ((self.references @ self.planes[:, cols]) >= 0) @ self.weights
            order = np.argsort(keys, kind="stable")
            counts = np.bincount(keys, minlength=2**bits)
            self.buckets.append((order, np.r_[0, np.cumsum(counts)]))

    def search(self, queries: np.ndarray, threshold: float | None = None) -> Matches:
        """Hash a batch of unit vectors, score unique candidates and apply an optional cosine cutoff."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.references.shape[1]:
            raise ValueError("Query and reference dimensions must match")
        projections = queries @ self.planes
        keys = ((projections.reshape(len(queries), self.tables, self.bits)) >= 0) @ self.weights
        tracks = np.full(len(queries), -1, dtype=np.int64)
        scores = np.full(len(queries), -np.inf, dtype=np.float32)
        counts = np.zeros(len(queries), dtype=np.int64)
        for i, (query, table_keys) in enumerate(zip(queries, keys, strict=True)):
            rows = np.unique(np.concatenate([
                order[offsets[key] : offsets[key + 1]]
                for (order, offsets), key in zip(self.buckets, table_keys, strict=True)
            ]))
            counts[i] = len(rows)
            if len(rows):
                similarities = self.references[rows] @ query
                best = similarities.argmax()
                scores[i] = similarities[best]
                if threshold is None or scores[i] >= threshold:
                    tracks[i] = self.track_ids[rows[best]]
        return Matches(tracks, scores, counts)
