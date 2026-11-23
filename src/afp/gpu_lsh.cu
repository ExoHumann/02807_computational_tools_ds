// One block scans a stripe of a hash table for one query. Each warp scores
// a contiguous 256-float window without copying it into a candidate matrix.
__device__ inline bool wins(float score, int row, float best, int best_row) {
    return score > best || (score == best && row >= 0 && (best_row < 0 || row < best_row));
}

extern "C" __global__ void scan_buckets(
    const float* queries, const float* projections, const float* refs,
    const int* order, const int* offsets, const int nrefs, const int bits,
    const int tables, const int probes, const int stripes,
    float* scores, int* rows, int* visits
) {
    const int part = blockIdx.x;
    const int t = part / stripes, stripe = part % stripes, q = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int warps = blockDim.x / 32;
    __shared__ int starts[4], ends[4], winners[8];
    __shared__ float maxima[8];
    if (threadIdx.x == 0) {
        const float* p = projections + (q * tables + t) * bits;
        int key = 0, first = -1, second = -1;
        for (int b = 0; b < bits; ++b) {
            if (p[b] >= 0) key |= 1 << b;
            if (first < 0 || fabsf(p[b]) < fabsf(p[first])) {
                second = first; first = b;
            } else if (second < 0 || fabsf(p[b]) < fabsf(p[second])) {
                second = b;
            }
        }
        const int masks[4] = {0, 1 << first, 1 << second, (1 << first) | (1 << second)};
        const int* boundaries = offsets + t * ((1 << bits) + 1);
        int counted = 0;
        for (int k = 0; k <= probes; ++k) {
            const int bucket = key ^ masks[k];
            const int lo = boundaries[bucket], size = boundaries[bucket + 1] - lo;
            starts[k] = lo + size * stripe / stripes;
            ends[k] = lo + size * (stripe + 1) / stripes;
            counted += ends[k] - starts[k];
        }
        visits[q * gridDim.x + part] = counted;
    }
    __syncthreads();
    float best = -__int_as_float(0x7f800000);
    int best_row = -1;
    for (int k = 0; k <= probes; ++k) {
        for (int j = starts[k] + warp; j < ends[k]; j += warps) {
            const int row = order[t * nrefs + j];
            float dot = 0;
            #pragma unroll
            for (int d = lane; d < 256; d += 32)
                dot = fmaf(refs[row * 256 + d], queries[q * 256 + d], dot);
            #pragma unroll
            for (int delta = 16; delta > 0; delta /= 2)
                dot += __shfl_down_sync(0xffffffff, dot, delta);
            if (lane == 0 && wins(dot, row, best, best_row)) { best = dot; best_row = row; }
        }
    }
    if (lane == 0) { maxima[warp] = best; winners[warp] = best_row; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < warps; ++w) {
            if (wins(maxima[w], winners[w], best, best_row)) { best = maxima[w]; best_row = winners[w]; }
        }
        scores[q * gridDim.x + part] = best;
        rows[q * gridDim.x + part] = best_row;
    }
}

extern "C" __global__ void finish(
    const float* scores, const int* rows, const int* visits,
    const int* tracks, int parts, int* out_track, float* out_score, int* out_visits
) {
    const int q = blockIdx.x, lane = threadIdx.x;
    float best = -__int_as_float(0x7f800000);
    int best_row = -1, count = 0;
    for (int i = lane; i < parts; i += 32) {
        int entry = q * parts + i;
        count += visits[entry];
        if (wins(scores[entry], rows[entry], best, best_row)) {
            best = scores[entry]; best_row = rows[entry];
        }
    }
    for (int d = 16; d; d /= 2) {
        float next = __shfl_down_sync(0xffffffff, best, d);
        int row = __shfl_down_sync(0xffffffff, best_row, d);
        count += __shfl_down_sync(0xffffffff, count, d);
        if (wins(next, row, best, best_row)) { best = next; best_row = row; }
    }
    if (lane == 0) {
        out_track[q] = best_row < 0 ? -1 : tracks[best_row];
        out_score[q] = best;
        out_visits[q] = count;
    }
}
