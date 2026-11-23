"""Measure the effect of enlarging a song catalogue using the same head, queries and vectors."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afp.gpu_retrieval import gpu_exact


def predict(references: cp.ndarray, tracks: cp.ndarray, queries: np.ndarray) -> np.ndarray:
    """Run cosine search in bounded GPU batches and transfer only the predicted tracks."""
    return np.concatenate(
        [gpu_exact(references, tracks, cp.asarray(queries[i : i + 256]))[0].get() for i in range(0, len(queries), 256)]
    )


def load_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a fingerprint archive and check the cosine-search contract."""
    with np.load(path, allow_pickle=False) as archive:
        vectors, tracks = archive["vectors"], archive["track_id"]
    if vectors.ndim != 2 or vectors.shape[1] != 256 or tracks.shape != (len(vectors),):
        raise ValueError(f"Unexpected fingerprint shape in {path}")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
    return vectors, tracks


def compare_catalogues(processed: Path, small_manifest: Path) -> dict:
    """Compare two nested catalogues while holding audio features and query membership fixed."""
    with small_manifest.open(encoding="utf-8-sig", newline="") as handle:
        small_ids = {int(row["track_id"]) for row in csv.DictReader(handle)}
    folder = processed / "fingerprints"
    references, track_ids = load_vectors(folder / "references.npz")
    subset = np.isin(track_ids, list(small_ids))
    if not subset.any() or subset.all():
        raise ValueError("The smaller catalogue must be a nonempty proper subset of the full catalogue.")
    small_tracks = track_ids[subset]
    gpu_refs, gpu_tracks = cp.asarray(references), cp.asarray(track_ids)
    gpu_subset = cp.asarray(subset)
    small_refs, small_gpu_tracks = gpu_refs[gpu_subset], gpu_tracks[gpu_subset]

    training_ids = set()
    shards = sorted((processed / "embeddings/pairs").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError("The run's training-pair shards are needed to check track separation.")
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            training_ids.update(np.unique(data["track_id"]).tolist())

    report = {
        "small_catalogue_tracks": np.unique(small_tracks).size,
        "small_catalogue_windows": len(small_refs),
        "medium_catalogue_tracks": np.unique(track_ids).size,
        "medium_catalogue_windows": len(references),
        "observed_training_tracks": len(training_ids),
        "queries": {},
        "sha256": {},
    }
    for name in ("queries", "queries_tiktok"):
        vectors, truth = load_vectors(folder / f"{name}.npz")
        if not training_ids.isdisjoint(truth.tolist()):
            raise ValueError(f"Training and evaluation share a track ID in {name}.")
        full_predictions = predict(gpu_refs, gpu_tracks, vectors)
        eligible = np.isin(truth, small_tracks)
        if not eligible.any():
            raise ValueError(f"No labelled queries for the smaller catalogue in {name}.")
        subset_predictions = predict(small_refs, small_gpu_tracks, vectors[eligible])
        small_hits = subset_predictions == truth[eligible]
        medium_hits = full_predictions[eligible] == truth[eligible]
        report["queries"][name] = {
            "total": len(truth),
            "absent_track_queries": int((~np.isin(truth, track_ids)).sum()),
            "medium_full_correct": int((full_predictions == truth).sum()),
            "paired_queries": int(eligible.sum()),
            "small_catalogue_correct": int(small_hits.sum()),
            "medium_catalogue_correct": int(medium_hits.sum()),
            "lost_to_larger_catalogue": int((small_hits & ~medium_hits).sum()),
            "gained_in_larger_catalogue": int((~small_hits & medium_hits).sum()),
        }
    for path in (folder / "references.npz", folder / "queries.npz", folder / "queries_tiktok.npz", small_manifest):
        with path.open("rb") as stream:
            report["sha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return report


def main() -> None:
    """Print the provenance and counts so the same paired result can be reproduced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path, help="Medium run's processed directory")
    parser.add_argument("--small-manifest", type=Path, default=Path("data/processed/manifest.csv"))
    args = parser.parse_args()
    print(json.dumps(compare_catalogues(args.processed, args.small_manifest), indent=2))


if __name__ == "__main__":
    main()
