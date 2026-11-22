"""Score the fixed exact and LSH methods against one catalogue with predetermined songs removed."""

import argparse
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd


def score(processed: Path, queries: Path, baseline: Path, output: Path) -> None:
    """Save scores for threshold calibration and track-level uncertainty analysis."""
    if not (baseline / "gpu_retrieval.py").is_file():
        raise FileNotFoundError("Provide the frozen PR #1 baseline directory (see reports/unknown-songs.md)")
    sys.path.insert(0, str(baseline.resolve()))
    from gpu_retrieval import GPUAngularLSH, gpu_exact

    with np.load(queries) as q:
        table = pd.DataFrame({k: q[k] for k in ("track_id", "known", "phase", "family", "distortion")})
        vectors = cp.asarray(q["vectors"])
    with np.load(processed / "fingerprints/references.npz") as ref:
        missing = table.loc[~table.known, "track_id"].unique()
        keep = ~np.isin(ref["track_id"], missing)
        tracks = ref["track_id"][keep]
        if not np.array_equal(np.isin(table.track_id, tracks), table.known):
            raise ValueError("Known/unknown labels disagree with actual catalogue membership")
        references, ids = cp.asarray(ref["vectors"][keep]), cp.asarray(tracks)
    index = GPUAngularLSH(references, ids, bits=10, tables=64, seed=2807)
    for name in ("exact", "lsh"):
        predictions, scores = [], []
        for start in range(0, len(table), 128):
            batch = vectors[start : start + 128]
            result = gpu_exact(references, ids, batch) if name == "exact" else index.search(batch, probes=3)
            predictions.extend(result[0].get())
            scores.extend(result[1].get())
        table[name + "_track"] = predictions
        table[name + "_score"] = scores
    if (table.lsh_score > table.exact_score + 2e-5).any():
        raise ValueError("Candidate score exceeds exhaustive maximum")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.9g")
    print(f"{len(table)} queries, {len(missing)} absent tracks, {len(tracks)} indexed windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/medium/data/processed"))
    parser.add_argument("--queries", type=Path, default=Path("data/medium/unknown-song-queries.npz"))
    parser.add_argument("--baseline", type=Path, default=Path("data/review-baseline"))
    parser.add_argument("--output", type=Path, default=Path("reports/unknown-song-scores.csv.gz"))
    args = parser.parse_args()
    score(args.processed, args.queries, args.baseline, args.output)
