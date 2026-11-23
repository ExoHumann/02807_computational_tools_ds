"""Verify five candidate recordings by matching local spectral peaks on the GPU."""

import csv
import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import rfft

FRESH = Path("data/large/fresh")
os.environ.setdefault("CUPY_CACHE_DIR", str(Path("data/medium/cupy-cache").resolve()))
RATE, WINDOW, HOP = 8000, 1024, 128
HANN = np.hanning(WINDOW).astype("f4")
LOCAL = threading.local()


def peaks(samples: np.ndarray, rate: int, *, gpu: bool = False) -> np.ndarray:
    """Extract the same spectral landmarks offline on CPU or online on CUDA."""
    if gpu:
        import cupy as xp
        from cupyx.scipy import ndimage, signal

        fft = xp.fft.rfft
    else:
        import numpy as xp
        from scipy import ndimage, signal

        fft = rfft
    x = xp.asarray(samples)
    if rate != RATE:
        x = signal.resample_poly(x, RATE, rate)
    frames = xp.lib.stride_tricks.sliding_window_view(x, WINDOW)[::HOP]
    mag = abs(fft(frames * xp.asarray(HANN), axis=1))
    log = xp.log(xp.maximum(mag, 1e-7))
    contrast = log - ndimage.uniform_filter1d(log, 23, axis=1)
    mask = (log == ndimage.maximum_filter(log, size=(9, 9))) & (contrast > 0.65)
    mask &= mag > xp.maximum(1e-7, mag.max() * 0.001)
    mask[:, :12] = False
    mask[:, 495:] = False
    t, k = xp.where(mask)
    if not len(t):
        return np.empty((0, 2), dtype="f4")
    left, mid, right = log[t, k - 1], log[t, k], log[t, k + 1]
    fraction = xp.clip(0.5 * (left - right) / xp.minimum(left - 2 * mid + right, -1e-7), -0.5, 0.5)
    frequency = 48 * xp.log2((k + fraction) * RATE / WINDOW / 64)
    packed = xp.column_stack((frequency, t, contrast[t, k] + 0.36 * log[t, k])).astype("f4")
    host = packed.get() if gpu else packed
    freq, times, strength = host.T
    order = np.lexsort((-strength, times // 16))
    grouped = times[order] // 16
    starts = np.maximum.accumulate(np.where(np.r_[True, grouped[1:] != grouped[:-1]], np.arange(len(times)), 0))
    take = order[np.arange(len(times)) - starts < 6]
    return np.ascontiguousarray(host[take[np.argsort(freq[take])], :2])


def reference_audio(tid: int) -> np.ndarray:
    """Read the original candidate excerpt and downsample it for the GPU."""
    from scipy.signal import resample_poly

    if not hasattr(LOCAL, "archive"):
        LOCAL.archive = zipfile.ZipFile("data/large/fma_large.zip")
    blob = LOCAL.archive.read(f"fma_large/{tid // 1000:03d}/{tid:06d}.mp3")
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return resample_poly(x.mean(axis=1), RATE, sr)


CUDA = r"""
extern "C" __global__ void align(const float2* query, int nq, const float2* refs,
    const int* boundaries, const int* rows, const float2* hypotheses, float* scores) {
    // Each block finds the strongest temporal alignment for one candidate and warp.
    __shared__ int votes[2048];
    __shared__ int maximum;
    int tid=threadIdx.x;
    if(tid==0) maximum=0;
    for(int i=tid;i<2048;i+=blockDim.x) votes[i]=0;
    __syncthreads();
    int group=rows[blockIdx.y];
    int start=boundaries[group], end=boundaries[group+1];
    float2 h=hypotheses[blockIdx.x];
    for(int i=tid;i<nq;i+=blockDim.x) {
        float2 p=query[i];
        float hz=p.x-h.y;
        int lo=start,hi=end;
        while(lo<hi) {
            int m=(lo+hi)/2;
            if(refs[m].x<hz-0.85f) lo=m+1; else hi=m;
        }
        for(int j=lo;j<end && refs[j].x<=hz+0.85f;j++) {
            int bin=__float2int_rn((refs[j].y-p.y*h.x)*0.5f)+700;
            if(bin>=1 && bin<2047) atomicAdd(votes+bin,1);
        }
    }
    __syncthreads();
    int best=0;
    for(int i=tid+1;i<2047;i+=blockDim.x) best=max(best,votes[i-1]+votes[i]+votes[i+1]);
    atomicMax(&maximum,best);
    __syncthreads();
    if(tid==0) scores[blockIdx.y*gridDim.x+blockIdx.x]=((float)maximum)/max(1,nq);
}
"""


class Verifier:
    """Keep reference peaks resident on GPU and evaluate the same warp grid for every query."""

    def __init__(self, ids, features):
        import cupy as cp

        self.lookup = {int(t): i for i, t in enumerate(ids)}
        self.refs = cp.asarray(np.concatenate(features))
        self.boundaries = cp.asarray(np.r_[0, np.cumsum([len(f) for f in features])].astype("i4"))
        grid = {(round(float(t), 4), float(p)) for t in np.arange(0.70, 1.501, 0.04) for p in range(-20, 21, 2)}
        grid.update((1.0, float(p)) for p in range(-20, 21, 2))
        grid.update((float(t), 48 * np.log2(t)) for t in np.arange(1.0, 1.501, 0.0125))
        self.grid = cp.asarray(sorted(grid), dtype="f4")
        self.kernel = cp.RawKernel(CUDA, "align")

    def match(self, q, candidates):
        """Return the best coherent peak vote for each proposed recording."""
        import cupy as cp

        rows = cp.asarray([self.lookup[int(k)] for k in candidates], dtype="i4")
        out = cp.empty((len(rows), len(self.grid)), dtype="f4")
        self.kernel(
            (len(self.grid), len(rows)),
            (128,),
            (cp.asarray(q), np.int32(len(q)), self.refs, self.boundaries, rows, self.grid, out),
        )
        return out.max(axis=1).get()


def query_audio():
    """Stream stored NumPy rows without allocating the full two-gigabyte matrix."""
    with zipfile.ZipFile(FRESH / "queries.npz") as archive, archive.open("audio.npy") as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError("Unexpected audio array format")
        (rows, width), fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if fortran or dtype != np.dtype("float32"):
            raise ValueError("Expected row-major float32 query audio")
        for _ in range(rows):
            yield np.frombuffer(stream.read(width * dtype.itemsize), dtype=dtype, count=width).copy()


def index_candidates(ids):
    """Extract reference features once for each track proposed in this benchmark."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:
        features = []
        for offset in range(0, len(ids), 128):
            decoded = pool.map(reference_audio, ids[offset : offset + 128])
            features.extend(peaks(x, RATE, gpu=True) for x in decoded)
            if offset % 1024 == 0:
                print(f"Prepared {len(features)}/{len(ids)} reference peak maps", flush=True)
    return features, time.perf_counter() - start


def prepare():
    """Overlap reference indexing with packaging of the already generated queries."""
    with np.load(FRESH / "shortlist.npz") as shortlist:
        ids = np.unique(shortlist["candidates"])
    features, seconds = index_candidates(ids)
    tmp = FRESH / "candidate-landmarks.tmp.npz"
    np.savez(
        tmp,
        track_id=ids,
        peaks=np.concatenate(features),
        offsets=np.r_[0, np.cumsum([len(x) for x in features])],
        seconds=seconds,
    )
    tmp.replace(FRESH / "candidate-landmarks.npz")


def main():
    """Time warm, one-clip-at-a-time verification and save the five independent scores."""
    import cupy as cp

    with np.load(FRESH / "queries.npz") as data:
        qs = {k: data[k] for k in data.files if k != "audio"}

    ids = np.unique(qs["candidates"])
    start = time.perf_counter()
    cached = FRESH / "candidate-landmarks.npz"
    if cached.exists():
        with np.load(cached) as saved:
            if not np.array_equal(ids, saved["track_id"]):
                raise ValueError("Cached candidate list differs. Rebuild with --prepare.")
            features = np.split(saved["peaks"], saved["offsets"][1:-1])
            offline_seconds = float(saved["seconds"])
    else:
        features, offline_seconds = index_candidates(ids)
    verifier = Verifier(ids, features)
    sr = int(qs["sample_rate"])
    verifier.match(peaks(next(query_audio()), sr, gpu=True), qs["candidates"][0])
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    print(f"Indexed {len(ids)} candidate tracks in {setup:.1f}s", flush=True)
    group_map = {}
    with (FRESH / "groups.csv").open() as f:
        for r in csv.DictReader(f):
            group_map[int(r["track_id"])] = int(r["group_id"])
    timings, landmarks = [], []
    for i, (sample, candidates) in enumerate(zip(query_audio(), qs["candidates"], strict=True)):
        begin = time.perf_counter()
        landmarks.append(verifier.match(peaks(sample, sr, gpu=True), candidates))
        timings.append((time.perf_counter() - begin) * 1000)
        if (i + 1) % 512 == 0:
            print(f"Verified {i + 1}/{len(qs['candidates'])} clips", flush=True)
    np.savez_compressed(
        "reports/landmark-scores.npz",
        **{
            k: qs[k] for k in ("phase", "family", "distortion", "track_id", "group_id", "known", "scores", "candidates")
        },
        candidate_groups=np.vectorize(group_map.__getitem__)(qs["candidates"]),
        landmarks=np.stack(landmarks),
        verifier_ms=timings,
    )
    details = {
        "indexed_candidate_tracks": len(ids),
        "candidate_peak_bytes": int(verifier.refs.nbytes),
        "indexed_seconds": offline_seconds,
        "initialisation_and_warmup_seconds": setup,
        "warp_hypotheses": len(verifier.grid),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "verifier_median_ms": float(np.median(timings)),
        "verifier_p95_ms": float(np.percentile(timings, 95)),
    }
    Path("reports/landmark-runtime.json").write_text(json.dumps(details, indent=2))
    print(details, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Precompute reference peaks for the saved shortlist.")
    if parser.parse_args().prepare:
        prepare()
    else:
        main()
