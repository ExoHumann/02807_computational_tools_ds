"""Time decoded-host-audio to decision with one live query per call on the RTX GPU."""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from fresh_shortlist import FRESH, REFERENCE, ROOT, load_cohort
from landmark_verify import RATE, Verifier, peaks, query_audio, reference_audio

os.environ.setdefault("HF_HOME", str(ROOT / "data/medium/hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "src"))


# One warp scores each visited reference window. Keep its highest cosine per recording
# group. Uint64 ordering also preserves earliest-row tie resolution within each group.
KERNEL = r"""
extern "C" __global__ void grouped(const float* q,const float* projections,const float* refs,
    const int* order,const int* offsets,const int* groups,int n,unsigned long long* best) {
    int table=blockIdx.x/16, probe=(blockIdx.x/4)%4, stripe=blockIdx.x%4;
    __shared__ int low,high;
    if(threadIdx.x==0) {
        const float* p=projections+table*10;
        int key=0, first=-1,second=-1;
        for(int b=0;b<10;b++){
            if(p[b]>=0) key|=1<<b;
            if(first<0||fabsf(p[b])<fabsf(p[first])) {second=first;first=b;}
            else if(second<0||fabsf(p[b])<fabsf(p[second])) second=b;
        }
        if(probe&1)key^=1<<first;
        if(probe&2)key^=1<<second;
        int begin=offsets[table*1025+key], count=offsets[table*1025+key+1]-begin;
        low=begin+count*stripe/4;high=begin+count*(stripe+1)/4;
    }
    __syncthreads();
    int lane=threadIdx.x%32;
    for(int j=low+threadIdx.x/32;j<high;j+=blockDim.x/32) {
        unsigned row=order[table*n+j];
        float dot=0;
        for(int d=lane;d<256;d+=32)dot=fmaf(q[d],refs[row*256+d],dot);
        for(int d=16;d>0;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);
        if(lane==0){
            unsigned bits=__float_as_uint(dot);
            unsigned sorted=(bits&0x80000000u)?~bits:(bits^0x80000000u);
            unsigned long long packed=((unsigned long long)sorted<<32)|(0xffffffffu-row);
            atomicMax(best+groups[row],packed);
        }
    }
}
"""


def main():
    """Alternate paired baseline and hybrid calls, including encoding and title selection."""
    import cupy as cp
    import torch

    from afp.embed import MuQBackbone
    from afp.fingerprint import load_head

    sys.path.insert(0, str(FRESH / "pinned_lsh"))
    from gpu_retrieval import GPUAngularLSH

    torch.set_num_threads(1)
    model = MuQBackbone(device="cuda").model.eval()
    head = load_head(ROOT / "data/medium/models/projection_head.pt").cuda().eval()
    with np.load(FRESH / "queries.npz") as raw:
        rate = int(raw["sample_rate"])
        chosen = np.sort(
            np.random.default_rng(280760).choice(np.flatnonzero(raw["phase"] == "test"), 64, replace=False)
        )
        chosen_set = set(chosen)
        samples = np.stack([x for i, x in enumerate(query_audio()) if i in chosen_set])
        expected = raw["candidate_group_id"][chosen]
        expected_tracks = raw["candidates"][chosen]
        expected_vec = raw["vectors"][chosen]
    mapping, cohort, _ = load_cohort()
    absent = {int(x["group_id"]) for x in cohort if not x["known"]}
    with np.load(REFERENCE) as file:
        ids = file["track_id"]
        labels = np.array([mapping[int(t)] for t in ids])
        keep = ~np.isin(labels, list(absent))
        ref_ids = ids[keep]
        index = GPUAngularLSH(cp.asarray(file["vectors"][keep]), cp.asarray(ref_ids), bits=10, tables=64, seed=2807)
    groups, codes = np.unique(labels[keep], return_inverse=True)
    codes = cp.asarray(codes, dtype="i4")
    kernel = cp.RawKernel(KERNEL, "grouped")
    keys = cp.zeros(len(groups), dtype="u8")

    def shortlist(vector):
        """Take the five highest-scoring distinct groups from the original LSH buckets."""
        keys.fill(0)
        projections = vector @ index.planes
        kernel(
            (64 * 4 * 4,),
            (128,),
            (vector, projections, index.refs, index.order, index.offsets, codes, np.int32(len(ref_ids)), keys),
        )
        selected = cp.argpartition(keys, -5)[-5:]
        ordered = cp.sort(keys[selected])[::-1]
        rows = (cp.uint64(0xFFFFFFFF) - (ordered & cp.uint64(0xFFFFFFFF))).astype("i4")
        bits = (ordered >> cp.uint64(32)).astype("u4")
        scores = cp.where(bits & cp.uint32(0x80000000), bits ^ cp.uint32(0x80000000), ~bits).view("f4")
        return index.tracks[rows].get(), scores.get()

    def embed(sample):
        """Encode one decoded waveform and transfer its unit vector without a CPU copy."""
        x = torch.from_numpy(sample[None, :]).cuda()
        y = model(x, output_hidden_states=True).last_hidden_state.mean(dim=1)
        return cp.from_dlpack(head(y.float()).contiguous())[0]

    rules = {
        r["method"]: r["rule"]
        for r in json.loads(Path("reports/landmark-results.json").read_text())["runs"]
        if r["target"] == 0.01
    }
    originals = []
    fresh_candidates = []
    with torch.inference_mode():
        for sample, vector in zip(samples, expected_vec, strict=True):
            originals.append(shortlist(cp.asarray(vector))[0])
            fresh_candidates.append(shortlist(embed(sample))[0])
    orig_groups = np.vectorize(mapping.__getitem__)(originals)
    live_groups = np.vectorize(mapping.__getitem__)(fresh_candidates)
    assert np.array_equal(orig_groups[:, 0], expected[:, 0]), "fast kernel and bulk top-one disagree"
    if not np.all(np.sort(orig_groups, axis=1) == np.sort(expected, axis=1)):
        raise ValueError("fast query kernel selected a different top-five set using the saved vectors")
    ids = np.unique(np.r_[np.ravel(fresh_candidates), expected_tracks.ravel()])
    verifier = Verifier(ids, [peaks(reference_audio(t), RATE, gpu=True) for t in ids])

    def decide(sample, method):
        """Predict a group, or abstain, starting from decoded 24 kHz host audio."""
        vector = embed(sample)
        tids, scores = shortlist(vector)
        rule = rules[method]
        if method == "fusion":
            acoustic = verifier.match(peaks(sample, rate, gpu=True), tids)
            scores = scores + rule["landmark_weight"] * acoustic
        order = np.argsort(-scores, kind="stable")
        best, second = order[:2]
        gap = float(scores[best]) - float(scores[second])
        return mapping[int(tids[best])] if float(scores[best]) >= rule["score"] and gap >= rule["gap"] else -1

    elapsed = {"cosine": [], "fusion": []}
    with torch.inference_mode():
        for method in elapsed:
            for sample in samples[:3]:
                decide(sample, method)
        for i, sample in enumerate(samples):
            times = {}
            for method in ("cosine", "fusion") if i % 2 == 0 else ("fusion", "cosine"):
                cp.cuda.Device().synchronize()
                start = time.perf_counter_ns()
                decide(sample, method)
                cp.cuda.Device().synchronize()
                times[method] = (time.perf_counter_ns() - start) / 1e6
            for method, durations in elapsed.items():
                durations.append(times[method])
    summary = {k: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))} for k, v in elapsed.items()}
    diff = np.array(elapsed["fusion"]) - elapsed["cosine"]
    summary.update(
        queries=len(samples),
        paired_median_extra_ms=float(np.median(diff)),
        same_saved_candidate_sets=int(np.all(np.sort(orig_groups, axis=1) == np.sort(expected, axis=1), axis=1).sum()),
        same_live_candidate_sets=int(np.all(np.sort(live_groups, axis=1) == np.sort(expected, axis=1), axis=1).sum()),
    )
    Path("reports/landmark-latency.json").write_text(json.dumps(summary, indent=2))
    with Path("reports/landmark-latency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_index", "cosine_ms", "fusion_ms"])
        w.writerows(zip(chosen, elapsed["cosine"], elapsed["fusion"], strict=True))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
