"""Speed-tolerant landmark search over the full FMA Large reference catalogue.

Each local three-frame constellation is hashed from its two signed log-frequency
intervals and the ratio of its temporal intervals. The hash is invariant to a
uniform pitch shift and playback-rate change. A GPU inverted index votes for
matching tracks; its compact source cache lives under ``data/large/fresh/acoustic``.

The index uses about 4 bytes per distinct (hash, track) posting plus a 4 MiB
GPU offset table. Query probes vary each quantized interval and timing ratio by
one bucket, and signatures with more than 256 reference tracks are ignored.
Numba accelerates catalogue encoding when installed; a NumPy fallback is kept
for environments that only have the project's declared dependencies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on minimal project installs
    njit = None


_BUCKET_FRAMES = 16
_FREQUENCY_STEP = 4.0
_KEY_COUNT = 1 << 20
_MAX_POSTINGS = 256
_CACHE_VERSION = "triplet-v1"
_PROBE_OFFSETS = np.asarray(
    [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
    dtype=np.int32,
)


def _encode_track(peaks: np.ndarray, track_row: int) -> np.ndarray:
    """Encode each three-cell local constellation as a packed hash/row pair."""
    count = len(peaks)
    if count < 3:
        return np.empty(0, dtype=np.uint64)

    order = np.argsort(peaks[:, 1])
    cells = np.empty(count, dtype=np.int32)
    low_frequency = np.empty(count, dtype=np.float32)
    high_frequency = np.empty(count, dtype=np.float32)
    low_time = np.empty(count, dtype=np.int32)
    high_time = np.empty(count, dtype=np.int32)
    groups = 0

    start = 0
    while start < count:
        first = int(order[start])
        cell = int(peaks[first, 1]) // _BUCKET_FRAMES
        end = start + 1
        while end < count and int(peaks[order[end], 1]) // _BUCKET_FRAMES == cell:
            end += 1

        low_row = first
        high_row = first
        for pos in range(start + 1, end):
            row = int(order[pos])
            if peaks[row, 0] < peaks[low_row, 0]:
                low_row = row
            if peaks[row, 0] > peaks[high_row, 0]:
                high_row = row

        cells[groups] = cell
        low_frequency[groups] = peaks[low_row, 0]
        high_frequency[groups] = peaks[high_row, 0]
        low_time[groups] = int(peaks[low_row, 1])
        high_time[groups] = int(peaks[high_row, 1])
        groups += 1
        start = end

    if groups < 3:
        return np.empty(0, dtype=np.uint64)

    packed = np.empty((groups - 2) * 8, dtype=np.uint64)
    used = 0
    for first in range(groups - 2):
        second = first + 1
        third = first + 2
        if cells[second] - cells[first] > 2 or cells[third] - cells[second] > 2:
            continue

        for choice in range(8):
            if choice & 1:
                f1, t1 = high_frequency[first], high_time[first]
            else:
                f1, t1 = low_frequency[first], low_time[first]
            if choice & 2:
                f2, t2 = high_frequency[second], high_time[second]
            else:
                f2, t2 = low_frequency[second], low_time[second]
            if choice & 4:
                f3, t3 = high_frequency[third], high_time[third]
            else:
                f3, t3 = low_frequency[third], low_time[third]

            first_interval = t2 - t1
            second_interval = t3 - t2
            if first_interval <= 0 or second_interval <= 0:
                continue

            first_delta = int(np.rint((f2 - f1) / _FREQUENCY_STEP))
            second_delta = int(np.rint((f3 - f2) / _FREQUENCY_STEP))
            first_delta = min(127, max(-128, first_delta)) + 128
            second_delta = min(127, max(-128, second_delta)) + 128
            time_ratio = int(np.rint(15 * first_interval / (first_interval + second_interval)))
            time_ratio = min(15, max(0, time_ratio))
            key = (first_delta << 12) | (second_delta << 4) | time_ratio
            packed[used] = (np.uint64(key) << np.uint64(32)) | np.uint64(track_row)
            used += 1

    return packed[:used]


if njit is not None:
    _encode_track = njit(cache=True)(_encode_track)


class AcousticIndex:
    """GPU inverted index over track-local acoustic landmark constellations.

    Args:
        ids: Eligible reference track IDs, one per row in ``offsets``.
        offsets: Peak-array offsets with length ``len(ids) + 1``.
        peaks: ``(P, 2)`` float array containing log frequency and STFT frame.

    The catalogue's sorted ``(hash, track)`` pairs are cached outside source
    control. The GPU representation is a 20-bit CSR inverted index. Search runs
    on CuPy's current stream so callers can use a nonblocking stream.
    """

    def __init__(self, ids: np.ndarray, offsets: np.ndarray, peaks: np.ndarray):
        import cupy as cp

        self.ids = np.ascontiguousarray(ids, dtype=np.int32)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int32)
        self.peaks = np.ascontiguousarray(peaks, dtype=np.float32)
        if self.peaks.ndim != 2 or self.peaks.shape[1] != 2:
            raise ValueError("peaks must have shape (P, 2)")
        if self.offsets.shape != (len(self.ids) + 1,):
            raise ValueError("offsets must have one more entry than ids")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.peaks):
            raise ValueError("offsets must span the complete peaks array")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be nondecreasing")

        pair_cache, count_cache = self._cache_paths()
        if pair_cache.exists() and count_cache.exists():
            packed = np.load(pair_cache, mmap_mode="r", allow_pickle=False)
            descriptor_counts = np.load(count_cache, allow_pickle=False)
        else:
            packed, descriptor_counts = self._build_catalogue()
            pair_cache.parent.mkdir(parents=True, exist_ok=True)
            self._save_array(pair_cache, packed)
            self._save_array(count_cache, descriptor_counts)

        packed_gpu = cp.asarray(packed, dtype=cp.uint64)
        key_gpu = (packed_gpu >> cp.uint64(32)).astype(cp.uint32)
        key_counts = cp.bincount(key_gpu, minlength=_KEY_COUNT)
        self._postings = packed_gpu.astype(cp.uint32)
        self._key_offsets = cp.empty(_KEY_COUNT + 1, dtype=cp.uint32)
        self._key_offsets[0] = 0
        self._key_offsets[1:] = cp.cumsum(key_counts, dtype=cp.uint64).astype(cp.uint32)
        self._track_ids = cp.asarray(self.ids, dtype=cp.int32)
        self._track_lengths = cp.asarray(np.maximum(descriptor_counts, 1), dtype=cp.float32)
        self._probe_offsets = cp.asarray(_PROBE_OFFSETS, dtype=cp.int32)
        self._num_tracks = len(self.ids)
        cp.cuda.get_current_stream().synchronize()

    def _cache_paths(self) -> tuple[Path, Path]:
        digest = hashlib.blake2b(digest_size=12)
        for values in (self.ids, self.offsets, self.peaks):
            digest.update(memoryview(values).cast("B"))
        cache_dir = Path(__file__).resolve().parents[1] / "data/large/fresh/acoustic"
        stem = f"{_CACHE_VERSION}-{digest.hexdigest()}"
        return cache_dir / f"{stem}-pairs.npy", cache_dir / f"{stem}-counts.npy"

    @staticmethod
    def _save_array(path: Path, values: np.ndarray) -> None:
        temporary = path.with_name(path.stem + ".tmp.npy")
        with temporary.open("wb") as target:
            np.save(target, values, allow_pickle=False)
        temporary.replace(path)

    def _build_catalogue(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = []
        descriptor_counts = np.zeros(len(self.ids), dtype=np.int32)
        for row, (start, stop) in enumerate(zip(self.offsets[:-1], self.offsets[1:], strict=True)):
            packed = _encode_track(self.peaks[start:stop], row)
            descriptor_counts[row] = len(packed)
            if len(packed):
                chunks.append(packed)
        if not chunks:
            return np.empty(0, dtype=np.uint64), descriptor_counts
        return np.unique(np.concatenate(chunks)), descriptor_counts

    def search(self, query_peaks: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """Return up to ``k`` track IDs and normalized GPU vote scores."""
        import cupy as cp

        if k <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        query = np.asarray(query_peaks, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("query_peaks must have shape (Q, 2)")
        query_pairs = _encode_track(query, 0)
        if not len(query_pairs):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        query_keys = np.unique((query_pairs >> np.uint64(32)).astype(np.uint32))
        if not len(query_keys):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        signatures = cp.asarray(query_keys, dtype=cp.uint32)
        delta = self._probe_offsets
        first = ((signatures >> cp.uint32(12)) & cp.uint32(255)).astype(cp.int32)
        second = ((signatures >> cp.uint32(4)) & cp.uint32(255)).astype(cp.int32)
        ratio = (signatures & cp.uint32(15)).astype(cp.int32)
        first_probe = first[:, None] + delta[None, :, 0]
        second_probe = second[:, None] + delta[None, :, 1]
        ratio_probe = ratio[:, None] + delta[None, :, 2]
        valid = (
            (first_probe >= 0)
            & (first_probe <= 255)
            & (second_probe >= 0)
            & (second_probe <= 255)
            & (ratio_probe >= 0)
            & (ratio_probe <= 15)
        )
        first_probe = cp.clip(first_probe, 0, 255).astype(cp.uint32)
        second_probe = cp.clip(second_probe, 0, 255).astype(cp.uint32)
        ratio_probe = cp.clip(ratio_probe, 0, 15).astype(cp.uint32)
        probes = (first_probe << cp.uint32(12)) | (second_probe << cp.uint32(4)) | ratio_probe
        starts = self._key_offsets[probes]
        counts = self._key_offsets[probes + cp.uint32(1)] - starts
        counts = cp.where(valid & (counts <= _MAX_POSTINGS), counts, 0).astype(cp.uint32)
        order = cp.argsort(counts, axis=1)
        counts = cp.take_along_axis(counts, order, axis=1).reshape(-1)
        starts = cp.take_along_axis(starts, order, axis=1).reshape(-1)
        total = int(cp.sum(counts, dtype=cp.uint64).get())
        if total == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        query_rows = cp.repeat(cp.arange(len(query_keys), dtype=cp.uint32), 27)
        query_rows = cp.repeat(query_rows, counts)
        posting_starts = cp.repeat(starts, counts)
        segment_starts = cp.repeat(cp.cumsum(counts, dtype=cp.uint64) - counts, counts)
        posting_counts = cp.repeat(counts, counts)
        posting_positions = cp.arange(total, dtype=cp.uint64) - segment_starts
        reference_rows = self._postings[posting_starts + posting_positions.astype(cp.uint32)]
        pairs = (query_rows.astype(cp.uint64) << cp.uint64(32)) | reference_rows.astype(cp.uint64)
        unique_pairs, first_match = cp.unique(pairs, return_index=True)
        reference_rows = unique_pairs.astype(cp.uint32)
        matched_df = posting_counts[first_match].astype(cp.float32)
        rarity = cp.log1p(cp.float32(self._num_tracks) / matched_df)
        weighted = cp.bincount(reference_rows, weights=rarity, minlength=self._num_tracks)
        votes = cp.bincount(reference_rows, minlength=self._num_tracks)
        scores = weighted.astype(cp.float32) / cp.sqrt(self._track_lengths)
        scores = cp.where(votes >= 2, scores, 0)

        take = min(int(k), self._num_tracks)
        best = cp.argpartition(scores, self._num_tracks - take)[-take:]
        best = best[cp.argsort(scores[best])[::-1]]
        best_scores = scores[best].get()
        best_ids = self._track_ids[best].get()
        keep = best_scores > 0
        return best_ids[keep].astype(np.int32, copy=False), best_scores[keep].astype(np.float32, copy=False)
