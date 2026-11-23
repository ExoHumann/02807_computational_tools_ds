"""GPU search and independent scoring for the two candidate sources on FMA Large."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from acoustic_baseline import FRESH, ROOT, groups, support

support()
from acoustic_catalogue import load_catalogue
from landmark_latency import KERNEL
from landmark_verify import Verifier, peaks, query_audio

os.environ.setdefault("CUPY_CACHE_DIR", str(FRESH / "cupy-cache"))


class NeuralIndex:
    """Return any number of highest-scoring distinct recording groups from PR 9's LSH buckets."""

    def __init__(self):
        import cupy as cp

        sys.path.insert(0, str(FRESH / "pinned_lsh"))
        from gpu_retrieval import GPUAngularLSH

        self.mapping, absent = groups()
        with np.load(ROOT / "data/large/references.npz") as d:
            track_ids = d["track_id"]
            keep = ~np.isin([self.mapping[int(t)] for t in track_ids], list(absent))
            ids = track_ids[keep]
            self.lsh = GPUAngularLSH(cp.asarray(d["vectors"][keep]), cp.asarray(ids), bits=10, tables=64, seed=2807)
        labels, inv = np.unique([self.mapping[int(t)] for t in ids], return_inverse=True)
        self.codes = cp.asarray(inv, dtype="i4")
        self.keys = cp.zeros(len(labels), dtype="u8")
        self.kernel = cp.RawKernel(KERNEL, "grouped")
        unique, counts = np.unique(ids, return_counts=True)
        order = np.argsort(ids, kind="stable")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        positions = np.arange(int(counts.max()))
        rows = order[np.minimum(starts[:, None] + positions, len(ids) - 1)]
        self.track_windows = cp.asarray(np.where(positions < counts[:, None], rows, -1), dtype="i4")
        self.track_index = np.full(int(unique.max()) + 1, -1, dtype="i4")
        self.track_index[unique] = np.arange(len(unique))

    def shortlist(self, vector, k):
        """GPU group-wise maximum cosine among visited windows, then top k groups."""
        import cupy as cp

        self.keys.fill(0)
        projections = vector @ self.lsh.planes
        self.kernel(
            (1024,),
            (128,),
            (
                vector,
                projections,
                self.lsh.refs,
                self.lsh.order,
                self.lsh.offsets,
                self.codes,
                np.int32(len(self.lsh.refs)),
                self.keys,
            ),
        )
        selected = cp.argpartition(self.keys, -k)[-k:]
        ordered = cp.sort(self.keys[selected])[::-1]
        rows = (cp.uint64(0xFFFFFFFF) - (ordered & cp.uint64(0xFFFFFFFF))).astype("i4")
        bits = (ordered >> cp.uint64(32)).astype("u4")
        values = cp.where(bits & cp.uint32(0x80000000), bits ^ cp.uint32(0x80000000), ~bits).view("f4")
        return self.lsh.tracks[rows].get(), values.get()

    def score(self, vector, tracks):
        """Score all reference windows of tracks nominated independently of neural LSH."""
        import cupy as cp

        indices = self.track_index[np.asarray(tracks, dtype="i4")]
        assert np.all(indices >= 0)
        rows = self.track_windows[cp.asarray(indices)]
        dots = cp.sum(self.lsh.refs[rows] * vector, axis=-1)
        return cp.where(rows >= 0, dots, -cp.inf).max(axis=1).get()


def distinct(tracks, mapping, k):
    """Keep the earliest representative of each proposed group, before truncation."""
    seen = set()
    indices = []
    for i, t in enumerate(tracks):
        g = mapping[int(t)]
        if g not in seen:
            indices.append(i)
            seen.add(g)
        if len(indices) == k:
            break
    return indices


def acoustic_source(index, q):
    """Return raw acoustic nominations without access to LSH or ground truth."""
    ids, _ = index.search(q, k=16)
    return np.asarray(ids, dtype="i4")


class AlignedNominations:
    """Verify a wider acoustic-only pool before nominating recording groups independently of LSH."""

    def __init__(self, index, verifier):
        self.index = index
        self.verifier = verifier

    def search(self, query_peaks, k=16):
        """Rank up to 64 acoustic candidates by their strongest coherent pitch/time alignment."""
        tracks, _ = self.index.search(query_peaks, k=64)
        if not len(tracks):
            return tracks, np.empty(0, dtype="f4")
        evidence = self.verifier.match(query_peaks, tracks)
        order = np.argsort(-evidence, kind="stable")[:k]
        return tracks[order], evidence[order]


def choose_generator(indices, mapping):
    """Choose the nomination method using only known development clips and original candidate misses."""
    from acoustic_baseline import previous_scores

    base = previous_scores()
    use = (base["phase"] == "development") & base["known"]
    missed = ~(base["candidate_groups"] == base["group_id"][:, None]).any(1)
    summary = {k: {"misses_recovered": 0, "acoustic_top5": 0} for k in indices}
    sr = int(np.load(FRESH / "queries-embedded.npz")["sample_rate"])
    for i, sample in enumerate(query_audio()):
        if not use[i]:
            continue
        q = peaks(sample, sr, gpu=True)
        for k, index in indices.items():
            proposed = acoustic_source(index, q)
            chosen = proposed[distinct(proposed, mapping, 5)]
            matched = any(mapping[int(t)] == base["group_id"][i] for t in chosen)
            summary[k]["acoustic_top5"] += int(matched)
            summary[k]["misses_recovered"] += int(matched and missed[i])
    winner = max(summary, key=lambda m: (summary[m]["misses_recovered"], summary[m]["acoustic_top5"]))
    decision = {
        "development_known": int(use.sum()),
        "development_misses": int((use & missed).sum()),
        "methods": summary,
        "selected": winner,
    }
    Path("reports/acoustic-selection.json").write_text(json.dumps(decision, indent=2))
    print("Development selection", decision, flush=True)
    return indices[winner]


def load_indexes():
    """Build the full acoustic, neural and verification indexes in the same eligible catalogue."""
    from acoustic_index import AcousticIndex
    from acoustic_offsets import OffsetIndex

    ids, offsets, all_peaks = load_catalogue()
    print(f"Loaded {len(ids):,} tracks and {len(all_peaks):,} peaks", flush=True)
    indices = {"triplet": AcousticIndex(ids, offsets, all_peaks), "offset_pairs": OffsetIndex(ids, offsets, all_peaks)}
    verifier = Verifier(ids, np.split(all_peaks, offsets[1:-1]))
    indices["offset_pairs_aligned"] = AlignedNominations(indices["offset_pairs"], verifier)
    acoustic = choose_generator(indices, groups()[0])
    print("Acoustic candidate index ready", flush=True)
    return NeuralIndex(), acoustic, verifier


def main():
    """Generate decisions' input scores for all 2,048 frozen query waveforms."""
    import cupy as cp
    from acoustic_baseline import previous_scores

    base = previous_scores()
    with np.load(FRESH / "queries-embedded.npz") as saved:
        vectors = saved["vectors"]
        sr = int(saved["sample_rate"])
        np.testing.assert_array_equal(saved["track_id"], base["track_id"])
    neural, acoustic, verifier = load_indexes()
    out = {}
    for tag in ("lsh10", "dual"):
        for name, dtype, fill in (("ids", "i4", -1), ("groups", "i4", -1), ("cos", "f4", -1e6), ("lm", "f4", 0)):
            out[f"{tag}_{name}"] = np.full((len(vectors), 10), fill, dtype=dtype)
    out["acoustic_groups"] = np.full((len(vectors), 5), -1, dtype="i4")
    for i, (sample, vector) in enumerate(zip(query_audio(), vectors, strict=True)):
        vec = cp.asarray(vector)
        q = peaks(sample, sr, gpu=True)
        tids, cos = neural.shortlist(vec, 10)
        primary = base["candidates"][i].tolist()
        primary_groups = set(base["candidate_groups"][i])
        extra = [j for j, t in enumerate(tids) if neural.mapping[int(t)] not in primary_groups][:5]
        wide = primary + [int(tids[j]) for j in extra]
        assert len(wide) == 10
        independent = acoustic_source(acoustic, q)
        independent = independent[distinct(independent, neural.mapping, 5)]
        acoustic_groups = [neural.mapping[int(t)] for t in independent]
        out["acoustic_groups"][i, : len(independent)] = acoustic_groups
        added = [int(t) for t, g in zip(independent, acoustic_groups, strict=True) if g not in primary_groups]
        dual = primary + added
        new = list(dict.fromkeys(wide[5:] + added))
        landmarks = dict(zip(primary, base["landmarks"][i], strict=True))
        landmarks.update(zip(new, verifier.match(q, new), strict=True))
        for tag, tracks, extra_scores in (
            ("lsh10", wide, cos[extra]),
            ("dual", dual, neural.score(vec, added) if added else []),
        ):
            n = len(tracks)
            out[f"{tag}_ids"][i, :n] = tracks
            out[f"{tag}_groups"][i, :n] = [neural.mapping[t] for t in tracks]
            out[f"{tag}_cos"][i, :n] = np.r_[base["scores"][i], extra_scores]
            out[f"{tag}_lm"][i, :n] = [landmarks[t] for t in tracks]
        if (i + 1) % 128 == 0:
            print(f"Scored {i + 1}/{len(vectors)} clips", flush=True)
    for k in ("track_id", "group_id", "known", "phase", "family", "distortion"):
        out[k] = base[k]
    np.savez_compressed("reports/acoustic-nominations.npz", **out)
    print("Saved candidate scores", flush=True)
    from acoustic_evaluation import main as evaluate
    from acoustic_timing import benchmark

    evaluate()
    benchmark(neural, acoustic, verifier)


if __name__ == "__main__":
    main()
