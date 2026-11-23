"""Speed-probed landmark pairs with catalogue-wide GPU voting for recording and time offset."""

from __future__ import annotations

import itertools

import numpy as np

# One posting stores a 20-bit catalogue row and a reference frame modulo 4096.
# Keys keep two absolute log-frequency bins and the integer frame interval.
KERNELS = r"""
extern "C" __global__ void build(const ushort2* peaks, const unsigned* rows,
    const int* offsets, int start, int stop, unsigned* cursors, unsigned* posts, int fill) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stop) return;
    unsigned row = rows[i];
    ushort2 a = peaks[i];
    for (int j = i + 1; j < offsets[row + 1]; ++j) {
        ushort2 b = peaks[j];
        int dt = b.y - a.y;
        if (dt > 42) break;
        if (dt < 9) continue;
        unsigned key = ((unsigned(a.x) * 512 + b.x) * 48) + dt;
        unsigned at = atomicAdd(cursors + key, 1u);
        if (fill) posts[at] = (row << 12) | (unsigned(a.y) & 4095u);
    }
}

extern "C" __global__ void vote(const unsigned* keys, const short* times,
    const unsigned* offsets, const unsigned* posts, unsigned* hist) {
    int term = blockIdx.x;
    unsigned lo = offsets[keys[term]], hi = offsets[keys[term]+1];
    // Limit common terms, which mostly vote for unrelated recordings.
    if (hi == lo) return;
    for (unsigned j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        unsigned post = posts[j];
        int offset = int(post & 4095) - int(times[term]);
        int bin = ((offset + 256) & 4095) >> 5;
        atomicAdd(hist + (post >> 12)*128 + bin, 1u);
    }
}

extern "C" __global__ void reduce(const unsigned* hist, unsigned* best, int n) {
    int row = blockIdx.x, lane = threadIdx.x;
    if (row >= n) return;
    __shared__ unsigned vals[128];
    unsigned a = hist[row*128 + lane];
    unsigned b = hist[row*128 + ((lane+1) & 127)];
    vals[lane] = a + b;
    __syncthreads();
    for (int d = 64; d > 0; d >>= 1) {
        if (lane < d) vals[lane] = max(vals[lane], vals[lane+d]);
        __syncthreads();
    }
    if (!lane) best[row] = vals[0];
}
"""


class OffsetIndex:
    """Probe plausible playback rates and pool track votes at compatible time offsets."""

    def __init__(self, ids: np.ndarray, offsets: np.ndarray, peaks: np.ndarray):
        import cupy as cp

        self.ids = ids
        self.rates = np.arange(0.80, 1.501, 0.025, dtype="f4")
        self.shifts = 48 * np.log2(self.rates)
        nkeys = 512 * 512 * 48
        if len(ids) >= (1 << 20) or np.max(peaks[:, 1]) >= 65536:
            raise ValueError("Catalogue exceeds packed posting bounds")
        sorted_peaks = np.empty_like(peaks, dtype="u2")
        for a, b in itertools.pairwise(offsets):
            part = peaks[a:b]
            by_time = part[np.argsort(part[:, 1], kind="stable")]
            sorted_peaks[a:b, 0] = np.clip(np.rint(by_time[:, 0]), 0, 511)
            sorted_peaks[a:b, 1] = by_time[:, 1]
        rows = cp.repeat(cp.arange(len(ids), dtype="u4"), cp.asarray(np.diff(offsets)))
        values = cp.asarray(sorted_peaks)
        boundaries = cp.asarray(offsets, dtype="i4")
        build = cp.RawKernel(KERNELS, "build")
        self.vote = cp.RawKernel(KERNELS, "vote")
        self.reduce = cp.RawKernel(KERNELS, "reduce")
        counts = cp.zeros(nkeys + 1, dtype="u4")

        def process(cursors, posts, fill):
            """Keep kernel launches short enough for a display GPU."""
            for a in range(0, len(peaks), 262144):
                b = min(a + 262144, len(peaks))
                args = (values, rows, boundaries, np.int32(a), np.int32(b), cursors, posts, np.int32(fill))
                build(((b - a + 127) // 128,), (128,), args)

        process(counts, counts, False)
        self.offsets = cp.zeros(nkeys + 1, dtype="u4")
        total = int(counts.sum(dtype="u8").get())
        if total >= 2**32:
            raise ValueError("Posting offsets require more than 32 bits")
        self.offsets[1:] = cp.cumsum(counts[:-1], dtype="u4")
        self.df = counts.get()
        self.posts = cp.empty(total, dtype="u4")
        process(self.offsets.copy(), self.posts, True)
        cp.cuda.get_current_stream().synchronize()
        print(f"Offset index ready: {total:,} postings, {self.posts.nbytes / 2**30:.2f} GiB", flush=True)

    def search(self, query_peaks: np.ndarray, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Map query pairs to plausible reference speeds and count aligned GPU postings."""
        import cupy as cp

        empty = np.empty(0, dtype="i4")
        if len(query_peaks) < 2 or k <= 0:
            return empty, empty
        q = query_peaks[np.argsort(query_peaks[:, 1], kind="stable")]
        freq, time = q.T
        delta = time[None, :] - time[:, None]
        first, second = np.where((delta >= 6) & (delta <= 53))
        rate = self.rates[:, None]
        dt = np.rint(delta[first, second] * rate).astype("i4")
        bins = np.clip(np.rint(freq - self.shifts[:, None]), 0, 511).astype("u4")
        valid = (dt >= 9) & (dt <= 42)
        keys = (bins[:, first] * 512 + bins[:, second]) * 48 + dt
        df = self.df[np.clip(keys, 0, len(self.df) - 1)]
        # Sample informative keys throughout the clip, keeping the posting work bounded.
        valid &= (df > 0) & (df <= 16384)
        cost = np.where(valid, df, np.iinfo(np.uint32).max)
        selected = np.zeros(valid.shape, dtype=bool)
        for segment in np.unique(time[first].astype("i4") // 80):
            columns = np.flatnonzero(time[first].astype("i4") // 80 == segment)
            width = min(48, len(columns))
            take = np.argpartition(cost[:, columns], width - 1, axis=1)[:, :width]
            selected[np.arange(len(rate))[:, None], columns[take]] = True
        valid &= selected
        packed_keys = cp.asarray(keys[valid], dtype="u4")
        anchor_times = cp.asarray(np.rint(time[first] * rate)[valid], dtype="i2")
        if not len(packed_keys):
            return empty, empty
        hist = cp.zeros((len(self.ids), 128), dtype="u4")
        best = cp.empty(len(self.ids), dtype="u4")
        self.vote((len(packed_keys),), (64,), (packed_keys, anchor_times, self.offsets, self.posts, hist))
        self.reduce((len(self.ids),), (128,), (hist, best, np.int32(len(self.ids))))
        take = min(k, len(self.ids))
        # Resolve equal vote counts by ascending catalogue ID.
        packed = (best.astype("u8") << 32) | (np.uint64(0xFFFFFFFF) - cp.arange(len(best), dtype="u8"))
        top = cp.sort(cp.partition(packed, -take)[-take:])[::-1].get()
        score = (top >> 32).astype("u4")
        index = (np.uint64(0xFFFFFFFF) - (top & np.uint64(0xFFFFFFFF))).astype("i4")
        return self.ids[index[score > 0]], score[score > 0]
