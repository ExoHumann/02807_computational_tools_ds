"""Prepare resumable catalogue landmark shards, excluding the complete absent recording groups."""

import argparse
import io
import json
import os
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from acoustic_baseline import FRESH, groups, support
from scipy.signal import resample_poly

support()
from landmark_verify import HANN, HOP, RATE, WINDOW

os.environ.setdefault("CUPY_CACHE_DIR", str(FRESH / "cupy-cache"))
OUT = FRESH / "acoustic"
SHARDS = OUT / "peaks"


def eligible():
    """Return the complete eligible catalogue, once, in stable ID order."""
    mapping, absent = groups()
    return np.asarray(sorted(t for t, g in mapping.items() if g not in absent), dtype="i4")


def decode(archive, tid):
    """Prepare a mono reference using the same resampler as the original verifier."""
    blob = archive.read(f"fma_large/{tid // 1000:03d}/{tid:06d}.mp3")
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = (x[:, 0] + x[:, 1]) * 0.5 if x.shape[1] == 2 else x.mean(axis=1)
    return resample_poly(mono, RATE, sr)


def batch_peaks(waves):
    """Batch the earlier spectral peak operator, masking padded STFT frames."""
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter, uniform_filter1d

    sizes = np.array([len(x) for x in waves])
    counts = (sizes - WINDOW) // HOP + 1
    padded = np.zeros((len(waves), int(sizes.max())), dtype="f4")
    for i, x in enumerate(waves):
        padded[i, : len(x)] = x
    frames = cp.lib.stride_tricks.sliding_window_view(cp.asarray(padded), WINDOW, axis=1)[:, ::HOP, :]
    mag = cp.abs(cp.fft.rfft(frames * cp.asarray(HANN), axis=2))
    ok = cp.arange(mag.shape[1])[None, :] < cp.asarray(counts)[:, None]
    mag *= ok[:, :, None]
    log = cp.log(cp.maximum(mag, 1e-7))
    contrast = log - uniform_filter1d(log, 23, axis=2)
    mask = (
        (log == maximum_filter(log, size=(1, 9, 9)))
        & (contrast > 0.65)
        & (mag > cp.maximum(1e-7, mag.max(axis=(1, 2))[:, None, None] * 0.001))
    )
    mask[:, :, :12] = False
    mask[:, :, 495:] = False
    b, t, k = cp.where(mask)
    mid = log[b, t, k]
    left, right = log[b, t, k - 1], log[b, t, k + 1]
    fraction = cp.clip(0.5 * (left - right) / cp.minimum(left - 2 * mid + right, -1e-7), -0.5, 0.5)
    packed = (
        cp.stack((b, t, 48 * cp.log2((k + fraction) * RATE / WINDOW / 64), contrast[b, t, k] + 0.36 * mid), axis=1)
        .astype("f4")
        .get()
    )
    cuts = np.searchsorted(packed[:, 0], np.arange(1, len(waves)))
    features = []
    for row in np.split(packed, cuts):
        if len(row) == 0:
            features.append(np.empty((0, 2), dtype="f4"))
            continue
        _, times, freq, strength = row.T
        order = np.lexsort((-strength, times // 16))
        bins = times[order] // 16
        index = np.arange(len(row))
        start = np.maximum.accumulate(np.where(np.r_[True, bins[1:] != bins[:-1]], index, 0))
        keep = order[index - start < 6]
        keep = keep[np.argsort(freq[keep])]
        features.append(np.ascontiguousarray(row[keep, 2:0:-1]))
    return features


def save_shard(path, ids, features):
    """Publish only complete shards for safe resumption."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        ids=np.asarray(ids, dtype="i4"),
        offsets=np.r_[0, np.cumsum([len(x) for x in features])].astype("i4"),
        peaks=np.concatenate(features),
    )
    tmp.replace(path)


def build(batch, workers, first=0, stop=None):
    """Prefetch decoded audio while the GPU extracts landmarks for each batch."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    all_ids = eligible()
    with np.load(FRESH / "candidate-landmarks.npz") as saved:
        cached = dict(zip(saved["track_id"], np.split(saved["peaks"], saved["offsets"][1:-1]), strict=True))
    start = time.perf_counter()
    done = 0
    with zipfile.ZipFile("data/large/fma_large.zip") as archive, ThreadPoolExecutor(workers) as pool:
        for n in range(first, min(stop or len(all_ids), len(all_ids)), 512):
            ids = all_ids[n : n + 512]
            path = SHARDS / f"{n:06d}.npz"
            if path.exists():
                continue
            features = {int(t): cached[t] for t in ids if t in cached}
            need = iter([int(t) for t in ids if t not in cached])
            queue = deque()

            def prefetch(queue, need):
                while len(queue) < max(workers * 3, batch * 3):
                    t = next(need, None)
                    if t is None:
                        return
                    queue.append((t, pool.submit(decode, archive, t)))

            prefetch(queue, need)
            while queue:
                items = [queue.popleft() for _ in range(min(batch, len(queue)))]
                prefetch(queue, need)
                features.update(
                    zip((t for t, _ in items), batch_peaks([future.result() for _, future in items]), strict=True)
                )
            save_shard(path, ids, [features[t] for t in ids])
            done += len(ids)
            print(
                f"{n + len(ids)}/{len(all_ids)} indexed; {done / (time.perf_counter() - start):.1f} tracks/s; {time.perf_counter() - start:.1f}s",
                flush=True,
            )
    if first or stop:
        return
    (OUT / "complete.json").write_text(
        json.dumps(
            {"tracks": len(all_ids), "shards": len(list(SHARDS.glob("*.npz"))), "previously_cached": len(cached)}
        )
    )


def load_catalogue():
    """Load all peak arrays in the declared catalogue order with no missing IDs."""
    ids, counts, all_peaks = [], [], []
    for path in sorted(SHARDS.glob("??????.npz")):
        with np.load(path) as z:
            ids.append(z["ids"])
            counts.append(np.diff(z["offsets"]))
            all_peaks.append(z["peaks"])
    ids = np.concatenate(ids)
    np.testing.assert_array_equal(ids, eligible())
    return ids, np.r_[0, np.cumsum(np.concatenate(counts))].astype("i4"), np.concatenate(all_peaks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build(args.batch, args.workers)
