"""CUDA search for the project's 256-dimensional unit audio fingerprints."""

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np


@lru_cache(maxsize=1)
def _kernels() -> tuple:
    """Compile the bucket scanner and deterministic final reduction once per process."""
    source = Path(__file__).with_name("gpu_lsh.cu").read_text(encoding="utf-8")
    module = cp.RawModule(code=source)
    return module.get_function("scan_buckets"), module.get_function("finish")


def gpu_exact(references: cp.ndarray, tracks: cp.ndarray, queries: cp.ndarray) -> tuple:
    """Return device track IDs and scores via exhaustive FP32 cosine search, ties in row order."""
    similarities = queries @ references.T
    rows = similarities.argmax(axis=1)
    return tracks[rows], similarities[cp.arange(len(queries)), rows]


class GPUAngularLSH:
    """Random-hyperplane buckets with optional probes of the two nearest sign boundaries.

    References, index, queries and scores stay on the GPU. A window found by several tables
    can be scored more than once. The maximum and deterministic tie break are equivalent
    to reranking the unique union. Work is reported as window visits, including repeats.
    """

    def __init__(self, references: cp.ndarray, tracks: cp.ndarray, *, bits: int, tables: int, seed: int = 2807) -> None:
        """Build the index on-device from FP32 unit vectors and integer FMA track IDs."""
        if references.ndim != 2 or references.shape[1] != 256 or not len(references):
            raise ValueError("Expected a nonempty matrix of 256-dimensional fingerprints")
        if not 2 <= bits <= 16 or tables < 1 or tracks.shape != (len(references),):
            raise ValueError("Use 2–16 bits, at least one table and one track ID per window")
        self.refs = cp.ascontiguousarray(references, dtype=cp.float32)
        self.tracks = cp.ascontiguousarray(tracks, dtype=cp.int32)
        self.bits, self.tables = bits, tables
        # Seed on the host for compatibility with the NumPy index, then build all buckets in VRAM.
        planes = np.random.default_rng(seed).standard_normal((tables, bits, 256), dtype=np.float32)
        planes /= np.linalg.norm(planes, axis=2, keepdims=True)
        self.planes = cp.ascontiguousarray(cp.asarray(planes.reshape(-1, 256).T))
        self.order = cp.empty((tables, len(references)), dtype=cp.int32)
        self.offsets = cp.empty((tables, (1 << bits) + 1), dtype=cp.int32)
        weights = 1 << cp.arange(bits, dtype=cp.int32)
        for table in range(tables):
            signs = self.refs @ self.planes[:, table * bits : (table + 1) * bits] >= 0
            keys = (signs * weights).sum(axis=1, dtype=cp.int32)
            self.order[table] = cp.argsort(keys)
            self.offsets[table, 0] = 0
            self.offsets[table, 1:] = cp.bincount(keys, minlength=1 << bits).cumsum(dtype=cp.int32)
        self.scan, self.finish = _kernels()

    def search(self, queries: cp.ndarray, *, probes: int = 0) -> tuple:
        """Return device track IDs, scores, and window visits for a query batch.

        Probes 1 and 2 flip the smallest and next-smallest absolute projections. Probe 3 also
        flips both bits together. All tables visit their base bucket. An empty selection
        returns track -1 and score -inf.
        """
        if probes not in (0, 1, 2, 3) or queries.ndim != 2 or queries.shape[1] != 256:
            raise ValueError("Expected 256-dimensional queries and 0–3 extra probes per table")
        queries = cp.ascontiguousarray(queries, dtype=cp.float32)
        n = len(queries)
        track = cp.empty(n, dtype=cp.int32)
        score = cp.empty(n, dtype=cp.float32)
        count = cp.empty(n, dtype=cp.int32)
        if not n:
            return track, score, count
        projections = queries @ self.planes
        stripes = 4 if n == 1 else 1
        parts = self.tables * stripes
        local_scores = cp.empty((n, parts), dtype=cp.float32)
        local_rows = cp.empty((n, parts), dtype=cp.int32)
        local_counts = cp.empty((n, parts), dtype=cp.int32)
        params = tuple(map(np.int32, (len(self.refs), self.bits, self.tables, probes, stripes)))
        self.scan(
            (parts, n),
            (128,),
            (
                queries,
                projections,
                self.refs,
                self.order,
                self.offsets,
                *params,
                local_scores,
                local_rows,
                local_counts,
            ),
        )
        self.finish(
            (n,), (32,), (local_scores, local_rows, local_counts, self.tracks, np.int32(parts), track, score, count)
        )
        return track, score, count
