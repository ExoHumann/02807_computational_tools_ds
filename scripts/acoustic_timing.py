"""Paired single-query service latency from decoded waveform to final recording decision."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def benchmark(neural, acoustic, verifier):
    """Alternate four warm paths, including genuinely concurrent acoustic and neural GPU submission."""
    import cupy as cp
    import torch
    from acoustic_baseline import FRESH, ROOT
    from acoustic_evaluation import decisions
    from acoustic_search import acoustic_source, distinct
    from landmark_verify import peaks, query_audio

    from afp.embed import MuQBackbone
    from afp.fingerprint import load_head

    torch.set_num_threads(1)
    model = MuQBackbone(device="cuda").model.eval()
    head = load_head(ROOT / "data/medium/models/projection_head.pt").cuda().eval()
    with np.load(FRESH / "queries-embedded.npz") as raw:
        sr = int(raw["sample_rate"])
        selected = np.sort(
            np.random.default_rng(280760).choice(np.flatnonzero(raw["phase"] == "test"), 64, replace=False)
        )
    select = set(selected)
    samples = [x for i, x in enumerate(query_audio()) if i in select]
    rules = json.loads(Path("reports/acoustic-results.json").read_text())["methods"]
    stream = cp.cuda.Stream(non_blocking=True)

    def acoustic_branch(sample, nominate):
        with stream:
            q = peaks(sample, sr, gpu=True)
            if not nominate:
                return q, []
            proposals = acoustic_source(acoustic, q)
            return q, proposals[distinct(proposals, neural.mapping, 5)]

    def call(sample, method, pool):
        dual = method.startswith("dual")
        future = None if method.endswith("_serial") else pool.submit(acoustic_branch, sample, dual)
        x = torch.from_numpy(sample[None, :]).cuda()
        y = model(x, output_hidden_states=True).last_hidden_state.mean(dim=1)
        vec = cp.from_dlpack(head(y.float()).contiguous())[0]
        tracks, cos = neural.shortlist(vec, 10 if method == "lsh10" else 5)
        q, proposals = future.result() if future is not None else acoustic_branch(sample, dual)
        if dual:
            existing = {neural.mapping[int(t)] for t in tracks}
            added = [int(t) for t in proposals if neural.mapping[int(t)] not in existing]
            if added:
                cos = np.r_[cos, neural.score(vec, added)]
                tracks = np.r_[tracks, added]
        lm = verifier.match(q, tracks)
        rule = rules[method.removesuffix("_serial")]["0.01"]["rule"]
        evidence = cos + rule["weight"] * lm
        order = np.argsort(-evidence, kind="stable")
        first, second = order[:2]
        gap = float(evidence[first]) - float(evidence[second])
        answer = (
            neural.mapping[int(tracks[first])] if float(evidence[first]) >= rule["score"] and gap >= rule["gap"] else -1
        )
        return answer, {neural.mapping[int(t)] for t in tracks}

    methods = ["lsh5", "lsh10", "dual", "dual_serial", "lsh5_serial"]
    rng = np.random.default_rng(11314)
    durations = {m: [] for m in methods}
    predictions = {m: [] for m in methods}
    candidate_sets = {m: [] for m in methods}
    with torch.inference_mode(), ThreadPoolExecutor(1) as pool:
        for m in methods:
            for x in samples[:3]:
                call(x, m, pool)
        for i, x in enumerate(samples):
            for m in rng.permutation(methods):
                cp.cuda.Device().synchronize()
                start = time.perf_counter_ns()
                answer, predicted = call(x, m, pool)
                cp.cuda.Device().synchronize()
                durations[m].append((time.perf_counter_ns() - start) / 1e6)
                predictions[m].append(answer)
                candidate_sets[m].append(predicted)
    np.testing.assert_array_equal(predictions["dual"], predictions["dual_serial"])
    assert candidate_sets["dual"] == candidate_sets["dual_serial"]
    np.testing.assert_array_equal(predictions["lsh5"], predictions["lsh5_serial"])
    assert candidate_sets["lsh5"] == candidate_sets["lsh5_serial"]
    checks = {}
    with np.load("reports/acoustic-nominations.npz") as d:
        for m in methods[:3]:
            g, _, accept = decisions(d, m, rules[m]["0.01"]["rule"])
            expected = np.where(accept, g[:, 0], -1)[selected]
            pools = [set(row[row >= 0]) for row in g[selected]]
            checks[m] = {
                "decisions_equal": int(np.sum(expected == predictions[m])),
                "sets_equal": sum(a == b for a, b in zip(pools, candidate_sets[m], strict=True)),
            }
    summary = {
        "samples": len(samples),
        "gpu": torch.cuda.get_device_name(),
        "matched_batch_comparison": checks,
        "methods": {
            m: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))} for m, v in durations.items()
        },
        "paired_extra_ms": float(np.median(np.array(durations["dual"]) - durations["lsh5"])),
    }
    with Path("reports/acoustic-latency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", *methods])
        w.writerows(zip(selected, *(durations[m] for m in methods), strict=True))
    Path("reports/acoustic-latency.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    from acoustic_search import load_indexes

    benchmark(*load_indexes())
