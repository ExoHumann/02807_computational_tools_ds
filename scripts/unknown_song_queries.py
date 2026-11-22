"""Generate fresh known/unknown queries on FMA medium with the frozen medium projection head."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from afp.augment import CATEGORIES, TEST_CATEGORIES, random_augment
from afp.data import SAMPLE_RATE, SEGMENT_S, load_processed, load_speech_bank
from afp.embed import MuQBackbone
from afp.fingerprint import load_head, split_tracks

SEED = 6821
HEAD_SHA = "4a5fb50d5dea344c0fdc803e9c825b6dfbd39b5dc86bafdce96c261e63801c6f"


def fresh_tracks(processed: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw 2,048 searchable eval tracks absent from both previous query archives and all training pairs."""
    manifest = pd.read_csv(processed / "manifest.csv")
    train, eligible = split_tracks(manifest.track_id, 0.2, 0)
    fp = processed / "fingerprints"
    with np.load(fp / "references.npz") as data:
        eligible = np.intersect1d(eligible, data["track_id"])
    for name in ("queries", "queries_tiktok"):
        with np.load(fp / f"{name}.npz") as data:
            eligible = np.setdiff1d(eligible, data["track_id"])
    shards = sorted((processed / "embeddings/pairs").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError("Training-pair shards are required to verify the split")
    for shard in shards:
        with np.load(shard) as data:
            if not np.isin(data["track_id"], train).all():
                raise ValueError("Training pairs do not match the declared medium split")
    return manifest.set_index("track_id"), np.random.default_rng(SEED).choice(eligible, 2048, replace=False)


def generate(processed: Path, model: Path, backbone: str, output: Path, workers: int, expected_sha: str) -> None:
    """Encode one standard and one TikTok-like query per independent source track on CUDA."""
    with model.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha:
            raise ValueError("The provided head does not match its expected digest")
    manifest, tracks = fresh_tracks(processed)
    speech = load_speech_bank(processed / "speech_test.npy")
    encoder = MuQBackbone("cuda", repo=backbone)
    head = load_head(model).cuda().eval()
    samples = int(SAMPLE_RATE * SEGMENT_S)

    def prepare(i: int) -> tuple[list[np.ndarray], list[str]]:
        track = int(tracks[i])
        wave = load_processed(processed / manifest.at[track, "path"])
        rng = np.random.default_rng([SEED, track])
        clips, labels = [], []
        for categories in (CATEGORIES, TEST_CATEGORIES):
            audio, _, label = random_augment(wave, rng, SAMPLE_RATE, categories, speech)
            start = int(rng.integers(len(audio) - samples + 1))
            clips.append(audio[start : start + samples].copy())
            labels.append(label)
        return clips, labels

    vectors, labels = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for first in range(0, len(tracks), 32):
            chunk = list(pool.map(prepare, range(first, min(first + 32, len(tracks)))))
            windows = np.stack([clip for clips, _ in chunk for clip in clips])
            labels.extend(label for _, names in chunk for label in names)
            for start in range(0, len(windows), 16):
                features = encoder.encode(windows[start : start + 16])
                vectors.append(head(torch.from_numpy(features).cuda()).cpu().numpy())
    result = np.concatenate(vectors)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=2e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        vectors=result,
        track_id=np.repeat(tracks, 2),
        distortion=labels,
        family=np.tile(["standard", "tiktok"], len(tracks)),
        phase=np.repeat(["development", "test"], len(tracks)),
        known=np.repeat(np.arange(len(tracks)) % 2 == 0, 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/medium/data/processed"))
    parser.add_argument("--model", type=Path, default=Path("data/medium/models/projection_head.pt"))
    parser.add_argument("--backbone", required=True, help="Pinned local MuQ checkpoint directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/medium/unknown-song-queries.npz"))
    parser.add_argument("--expected-head-sha256", default=HEAD_SHA)
    args = parser.parse_args()
    generate(args.processed, args.model, args.backbone, args.output, args.workers, args.expected_head_sha256)
