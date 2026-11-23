"""Build recording groups from FMA Large audio before selecting fresh evaluation songs."""

import csv
import gzip
import io
import itertools
import json
import re
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, resample_poly

ROOT = Path("data/large/fresh")
LOCAL = threading.local()
ARCHIVE = "data/large/fma_large.zip"


def buckets(pairs, values):
    """Add all pairs in each bucket."""
    for ids in values:
        pairs.update(itertools.combinations(sorted(ids), 2))


def propose(ids, means):
    """Retrieve candidate aliases using independent metadata, archive and embedding cues."""
    eligible = set(ids.tolist())
    names, crcs = defaultdict(list), defaultdict(list)
    with zipfile.ZipFile("data/medium/data/raw/fma_metadata.zip") as archive:
        path = next(name for name in archive.namelist() if name.endswith("/tracks.csv"))
        with io.TextIOWrapper(archive.open(path), encoding="utf8") as source:
            rows = csv.reader(source)
            cols = list(zip(next(rows), next(rows), strict=True))
            artist, title = cols.index(("artist", "name")), cols.index(("track", "title"))
            next(rows)
            for row in rows:
                tid = int(row[0])
                key = tuple(re.sub(r"[\W_]+", "", row[i].casefold()) for i in (artist, title))
                if tid in eligible and all(key):
                    names[key].append(tid)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            stem = Path(member.filename).stem
            if stem.isdecimal() and int(stem) in eligible and member.filename.endswith(".mp3"):
                crcs[member.CRC, member.file_size].append(int(stem))
    pairs = set()
    buckets(pairs, names.values())
    buckets(pairs, crcs.values())
    vectors = torch.nn.functional.normalize(torch.from_numpy(means).cuda())
    for start in range(0, len(ids), 256):
        values, indices = (vectors[start : start + 256] @ vectors.T).topk(6)
        near = indices.cpu().numpy()
        for i, j in zip(*np.where(values.cpu().numpy() >= 0.90), strict=True):
            a, b = int(ids[start + i]), int(ids[near[i, j]])
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


@lru_cache(maxsize=350)
def audio(tid):
    """Read one clean excerpt at 2 kHz mono for alignment verification."""
    if not hasattr(LOCAL, "archive"):
        LOCAL.archive = zipfile.ZipFile(ARCHIVE)
    name = f"fma_large/{tid // 1000:03d}/{tid:06d}.mp3"
    x, sr = sf.read(io.BytesIO(LOCAL.archive.read(name)), dtype="float32", always_2d=True)
    return resample_poly(x.mean(axis=1), 2000, sr).astype(np.float32)


def verify(pair):
    """Require two distinct aligned high-correlation three-second passages."""
    a, b = map(audio, pair)
    width = 6000
    if min(len(a), len(b)) < 2 * width:
        return None
    energy = np.r_[0.0, np.cumsum(b, dtype=float)]
    squares = np.r_[0.0, np.cumsum(b * b, dtype=float)]
    variance = squares[width:] - squares[:-width] - (energy[width:] - energy[:-width]) ** 2 / width
    valid = variance > width * 1e-5
    offsets = []
    for t in range(0, len(a) - width + 1, width):
        snippet = a[t : t + width] - a[t : t + width].mean()
        power = float(snippet @ snippet)
        if power < width * 1e-5:
            continue
        sim = correlate(b, snippet, mode="valid", method="fft") / np.sqrt(np.maximum(variance, 1e-10) * power)
        sim[~valid] = 0
        k = int(sim.argmax())
        if sim[k] >= 0.95:
            offsets.append((k - t, float(min(sim[k], 1))))
            for prev, score in offsets[:-1]:
                if abs(prev - (k - t)) <= 20:
                    return (*pair, round(min(score, offsets[-1][1]), 5))
    return None


def main():
    """Group confirmed aliases, then sample previously unused recording families."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with np.load("data/large/references.npz") as ref:
        tids, vectors = ref["track_id"], ref["vectors"]
    order = np.argsort(tids, kind="stable")
    tids, vectors = tids[order], vectors[order]
    ids, starts, counts = np.unique(tids, return_index=True, return_counts=True)
    assert np.all(tids[1:] >= tids[:-1])
    means = (np.add.reduceat(vectors, starts, axis=0) / counts[:, None]).astype("f4")
    candidates = propose(ids, means)
    print(f"Verifying {len(candidates)} candidate pairs from {len(ids)} tracks", flush=True)
    parent = {int(t): int(t) for t in ids}

    def group(t):
        while t != parent[t]:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    edges = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, match in enumerate(pool.map(verify, candidates)):
            if match:
                a, b, _val = match
                x, y = sorted((group(a), group(b)))
                parent[y] = x
                edges.append(match)
            if (n + 1) % 1000 == 0:
                print(f"Verified {n + 1}/{len(candidates)} pairs, {len(edges)} linked", flush=True)
    with (ROOT / "groups.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "group_id"])
        w.writerows((t, group(t)) for t in ids)
    old = set()
    for path in ("data/medium/data/processed/manifest.csv", "data/processed/manifest.csv"):
        with open(path, encoding="utf8") as f:
            old.update(int(r["track_id"]) for r in csv.DictReader(f))
    with np.load("data/medium/runner-up-queries.npz") as previous:
        old.update(previous["track_id"].tolist())
    excluded = {group(t) for t in old if t in parent}
    available = sorted(set(map(group, ids)) - excluded)
    rng = np.random.default_rng(28071123)
    chosen = rng.choice(available, 1024, replace=False).reshape(2, 2, 256)
    with (ROOT / "cohort.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "group_id", "phase", "known"])
        for phase, slice_ in zip(("development", "test"), chosen, strict=True):
            for known, block in enumerate(slice_):
                w.writerows((t, t, phase, known) for t in block)
    with gzip.open("reports/recording-alias-edges.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_a", "track_b", "min_aligned_correlation"])
        w.writerows(edges)
    summary = {
        "searchable": len(ids),
        "candidate_pairs": len(candidates),
        "verified_edges": len(edges),
        "groups": len(set(map(group, ids))),
        "multi_track_groups": sum(n > 1 for n in Counter(map(group, ids)).values()),
        "eligible_fresh_groups": len(available),
        "groups_per_phase_and_status": 256,
        "seed": 28071123,
    }
    Path("reports/recording-groups.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
