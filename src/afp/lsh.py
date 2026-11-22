"""Random-hyperplane LSH for unit-normalised audio fingerprints."""

from dataclasses import dataclass
from typing import Literal

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
        planes /= np.linalg.norm(planes, axis=2, keepdims=True)
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

    def search(
        self,
        queries: np.ndarray,
        threshold: float | None = None,
        *,
        probe_budget: int = 0,
        rerank_backend: Literal["numpy", "numba"] = "numpy",
    ) -> Matches:
        """Hash a batch, optionally probe nearby buckets, rerank, and apply a cosine cutoff.

        ``probe_budget`` adds at most one alternate bucket per selected table. Tables
        with the smallest absolute query-to-unit-plane projection are selected first.
        These least-certain sign bits are the most likely single-bit hash errors.
        The optional Numba backend scores candidates in place without copying them.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.references.shape[1]:
            raise ValueError("Query and reference dimensions must match")
        if rerank_backend not in ("numpy", "numba"):
            raise ValueError("rerank_backend must be 'numpy' or 'numba'")
        if not 0 <= probe_budget <= self.tables:
            raise ValueError("probe_budget must be between zero and the number of tables")
        projections = (queries @ self.planes).reshape(len(queries), self.tables, self.bits)
        keys = (projections >= 0) @ self.weights
        if probe_budget:
            absolute_projections = np.abs(projections)
            nearest_bits = absolute_projections.argmin(axis=2)
            margins = np.take_along_axis(absolute_projections, nearest_bits[:, :, None], axis=2)[:, :, 0]
        tracks = np.full(len(queries), -1, dtype=np.int64)
        scores = np.full(len(queries), -np.inf, dtype=np.float32)
        counts = np.zeros(len(queries), dtype=np.int64)
        for i, (query, table_keys) in enumerate(zip(queries, keys, strict=True)):
            bucket_rows = []
            for table, (order, offsets) in enumerate(self.buckets):
                key = table_keys[table]
                bucket_rows.append(order[offsets[key] : offsets[key + 1]])
            if probe_budget:
                selected = np.argpartition(margins[i], probe_budget - 1)[:probe_budget]
                for table in selected:
                    alternate_key = table_keys[table] ^ self.weights[nearest_bits[i, table]]
                    order, offsets = self.buckets[table]
                    bucket_rows.append(order[offsets[alternate_key] : offsets[alternate_key + 1]])
            rows = np.unique(np.concatenate(bucket_rows))
            counts[i] = len(rows)
            if len(rows):
                if rerank_backend == "numba":
                    from afp.rerank import best_candidate

                    best_row, score = best_candidate(self.references, rows, query)
                    scores[i] = score
                else:
                    similarities = self.references[rows] @ query
                    best = similarities.argmax()
                    best_row = rows[best]
                    scores[i] = similarities[best]
                if threshold is None or scores[i] >= threshold:
                    tracks[i] = self.track_ids[best_row]
        return Matches(tracks, scores, counts)
