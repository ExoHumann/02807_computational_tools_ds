"""Produce fingerprints from the processed audio: cache MuQ features, train the projection head, encode.

Each step is a subcommand and is resumable:

    uv run python -m afp.fingerprint cache    # MuQ features: clean windows, training pairs, evaluation queries
    uv run python -m afp.fingerprint train    # contrastive training of the head on the cached pairs
    uv run python -m afp.fingerprint encode   # fingerprints for every window and query, plus exact-search top-1

Tracks are split into a training part (used only to build the pairs the head learns from) and an evaluation part
(the source of the queries), so the head is never scored on tracks it was trained on. All tracks are in the
reference database.

Outputs under ``data/processed``: ``embeddings/{clean,pairs,queries}/shard_*.npz`` (cached MuQ features) and
``fingerprints/{references,queries}.npz`` (the fingerprints). The trained head goes to ``models/projection_head.pt``.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from afp.augment import random_augment
from afp.data import SAMPLE_RATE, SEGMENT_S, load_processed, segment
from afp.embed import EMBED_DIM, HIDDEN_DIM, MuQBackbone, ProjectionHead, train_head

EMBEDDINGS_DIR = "embeddings"
FINGERPRINTS_DIR = "fingerprints"


def split_tracks(track_ids: pd.Series, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split track ids into a training part and an evaluation part.

    Args:
        track_ids: All track ids.
        train_frac: Fraction of tracks used to build training pairs.
        seed: Seed for the shuffle.

    Returns:
        A tuple ``(train_ids, eval_ids)`` of sorted arrays.
    """
    shuffled = np.random.default_rng(seed).permutation(np.asarray(track_ids))
    n_train = int(len(shuffled) * train_frac)
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def _embed(backbone: MuQBackbone, windows: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the backbone over ``windows`` in batches. Any object with a compatible ``encode`` method works."""
    batches = [backbone.encode(windows[i : i + batch_size]) for i in range(0, len(windows), batch_size)]
    return np.concatenate(batches)


def _save_shard(path: Path, **arrays: np.ndarray) -> None:
    """Write arrays to ``path`` atomically so an interrupted run never leaves a half-written shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "wb") as f:
        np.savez(f, **arrays)
    temporary.replace(path)


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    """Load and concatenate every ``shard_*.npz`` in a directory.

    Args:
        directory: Folder written by :func:`cache_clean` or :func:`cache_clips`.

    Returns:
        A dict mapping each array name to the concatenation across shards.

    Raises:
        FileNotFoundError: If the directory holds no shards.
    """
    files = sorted(Path(directory).glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {directory}, run the cache step first")
    parts = [dict(np.load(f)) for f in files]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def cache_clean(
    manifest: pd.DataFrame,
    processed_dir: Path,
    backbone: MuQBackbone,
    batch_size: int = 4,
    tracks_per_shard: int = 50,
) -> None:
    """Embed every clean 10 s window (5 s hop) of every track and save the features in shards.

    Finished shards are skipped. Delete ``embeddings/clean`` to start over with different settings.

    Args:
        manifest: Table with ``track_id`` and ``path`` (relative to ``processed_dir``).
        processed_dir: Directory holding the processed audio.
        backbone: Feature extractor.
        batch_size: Windows per backbone call.
        tracks_per_shard: Tracks per output file.
    """
    out_dir = Path(processed_dir) / EMBEDDINGS_DIR / "clean"
    for shard, first in enumerate(tqdm(range(0, len(manifest), tracks_per_shard), desc="clean windows")):
        path = out_dir / f"shard_{shard:04d}.npz"
        if path.exists():
            continue
        windows, track_id, segment_idx, start_s = [], [], [], []
        for row in manifest.iloc[first : first + tracks_per_shard].itertuples():
            segments, starts = segment(load_processed(Path(processed_dir) / row.path))
            windows.append(segments)
            track_id += [row.track_id] * len(segments)
            segment_idx += range(len(segments))
            start_s += starts.tolist()
        _save_shard(
            path,
            embeddings=_embed(backbone, np.concatenate(windows), batch_size).astype(np.float16),
            track_id=np.array(track_id, dtype=np.int32),
            segment_idx=np.array(segment_idx, dtype=np.int16),
            start_s=np.array(start_s, dtype=np.float32),
        )


def cache_clips(
    kind: str,
    manifest: pd.DataFrame,
    track_ids: np.ndarray,
    n_clips: int,
    processed_dir: Path,
    backbone: MuQBackbone,
    with_clean: bool,
    seed: int,
    batch_size: int = 4,
    clips_per_shard: int = 100,
) -> None:
    """Cut random 10 s clips, distort each one and save the MuQ features of the distorted (and optionally clean) clip.

    The whole track is distorted before the clip is cut, so filters and reverb have no edge effects and time
    stretching maps the clip back to its position in the original track. Shards are seeded individually, so an
    interrupted run can be repeated. Delete ``embeddings/<kind>`` to start over with different settings.

    Args:
        kind: Output folder name below ``embeddings``: ``"pairs"`` for training or ``"queries"`` for evaluation.
        manifest: Table with ``track_id`` and ``path`` (relative to ``processed_dir``).
        track_ids: Tracks the clips are drawn from.
        n_clips: Number of clips.
        processed_dir: Directory holding the processed audio.
        backbone: Feature extractor.
        with_clean: Also embed the undistorted clip at the same position (needed to train the head).
        seed: Base seed.
        batch_size: Windows per backbone call.
        clips_per_shard: Clips per output file.
    """
    paths = manifest.set_index("track_id")["path"]
    window = round(SEGMENT_S * SAMPLE_RATE)
    out_dir = Path(processed_dir) / EMBEDDINGS_DIR / kind
    for shard in tqdm(range(math.ceil(n_clips / clips_per_shard)), desc=kind):
        path = out_dir / f"shard_{shard:04d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng([seed, shard])
        count = min(clips_per_shard, n_clips - shard * clips_per_shard)
        clean, distorted, track_id, start_s, labels = [], [], [], [], []
        while len(distorted) < count:
            chosen = int(rng.choice(track_ids))
            track = load_processed(Path(processed_dir) / paths[chosen])
            start = int(rng.integers(0, len(track) - window + 1))
            audio, rate, label = random_augment(track, rng, SAMPLE_RATE)
            distorted_start = round(start / rate)
            if distorted_start + window > len(audio):
                continue
            clean.append(track[start : start + window])
            distorted.append(audio[distorted_start : distorted_start + window])
            track_id.append(chosen)
            start_s.append(start / SAMPLE_RATE)
            labels.append(label)
        arrays = {
            "distorted": _embed(backbone, np.stack(distorted), batch_size).astype(np.float16),
            "track_id": np.array(track_id, dtype=np.int32),
            "start_s": np.array(start_s, dtype=np.float32),
            "augmentation": np.array(labels),
        }
        if with_clean:
            arrays["clean"] = _embed(backbone, np.stack(clean), batch_size).astype(np.float16)
        _save_shard(path, **arrays)


def train(
    processed_dir: Path,
    model_path: Path,
    hidden_dim: int = HIDDEN_DIM,
    out_dim: int = EMBED_DIM,
    **train_kwargs,
) -> list[tuple[float, float]]:
    """Train the projection head on the cached pairs and save it.

    Args:
        processed_dir: Directory holding ``embeddings/pairs``.
        model_path: Where to save the head (a dict with the weights and layer sizes).
        hidden_dim: Hidden layer width.
        out_dim: Fingerprint size.
        **train_kwargs: Passed to :func:`afp.embed.train_head` (epochs, batch_size, lr, temperature, seed, device).

    Returns:
        A list of ``(train_loss, val_loss)`` per epoch.
    """
    pairs = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / "pairs")
    in_dim = pairs["clean"].shape[1]
    head = ProjectionHead(in_dim, hidden_dim, out_dim)
    history = train_head(head, pairs["clean"], pairs["distorted"], **train_kwargs)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": head.state_dict(), "in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim}, model_path
    )
    return history


def load_head(model_path: Path) -> ProjectionHead:
    """Load a head saved by :func:`train` in evaluation mode."""
    saved = torch.load(model_path)
    head = ProjectionHead(saved["in_dim"], saved["hidden_dim"], saved["out_dim"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


def _apply_head(head: ProjectionHead, embeddings: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    """Turn backbone features into unit-norm ``float32`` fingerprints."""
    head.to(device)
    with torch.no_grad():
        batches = [
            head(torch.from_numpy(embeddings[i : i + batch_size].astype(np.float32)).to(device)).cpu().numpy()
            for i in range(0, len(embeddings), batch_size)
        ]
    return np.concatenate(batches)


def top1_hit_rate(
    references: np.ndarray,
    reference_track: np.ndarray,
    queries: np.ndarray,
    query_track: np.ndarray,
    labels: np.ndarray,
    chunk: int = 256,
) -> pd.DataFrame:
    """Exact nearest-neighbour search: how often is the closest reference window from the query's source track?

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        reference_track: Track id of each reference fingerprint.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        query_track: Source track id of each query.
        labels: Distortion category of each query.
        chunk: Queries searched at a time, to bound memory.

    Returns:
        A table indexed by distortion category plus ``overall`` with the query count ``n`` and the hit rate ``top1``.
    """
    predicted = np.concatenate(
        [reference_track[(queries[i : i + chunk] @ references.T).argmax(axis=1)] for i in range(0, len(queries), chunk)]
    )
    hits = pd.DataFrame({"label": labels, "hit": predicted == query_track})
    table = hits.groupby("label")["hit"].agg(n="size", top1="mean")
    table.loc["overall"] = [len(hits), hits["hit"].mean()]
    return table


def encode(processed_dir: Path, model_path: Path, device: str | None = None) -> pd.DataFrame:
    """Turn the cached features into fingerprints, save them, and report exact-search top-1 per distortion.

    Args:
        processed_dir: Directory holding ``embeddings/clean`` and ``embeddings/queries``.
        model_path: Head saved by :func:`train`.
        device: Device for the head. Defaults to CUDA when available.

    Returns:
        The table from :func:`top1_hit_rate`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = load_head(model_path)
    out_dir = Path(processed_dir) / FINGERPRINTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / "clean")
    references = _apply_head(head, clean["embeddings"], device)
    np.savez(
        out_dir / "references.npz",
        vectors=references,
        track_id=clean["track_id"],
        segment_idx=clean["segment_idx"],
        start_s=clean["start_s"],
    )
    queries = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / "queries")
    query_vectors = _apply_head(head, queries["distorted"], device)
    np.savez(
        out_dir / "queries.npz",
        vectors=query_vectors,
        track_id=queries["track_id"],
        start_s=queries["start_s"],
        augmentation=queries["augmentation"],
    )
    return top1_hit_rate(references, clean["track_id"], query_vectors, queries["track_id"], queries["augmentation"])


def main(argv: list[str] | None = None) -> None:
    """Command line entry point with the subcommands ``cache``, ``train`` and ``encode``."""
    parser = argparse.ArgumentParser(description="Produce fingerprints from the processed audio.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/projection_head.pt"))
    parser.add_argument("--seed", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache", help="extract and store MuQ features")
    cache.add_argument("--limit-tracks", type=int, default=None, help="use only the first N tracks of the manifest")
    cache.add_argument("--train-frac", type=float, default=0.2, help="fraction of tracks used for training pairs")
    cache.add_argument("--pairs", type=int, default=20_000, help="number of training pairs")
    cache.add_argument("--queries", type=int, default=3_000, help="number of evaluation queries")
    cache.add_argument("--batch-size", type=int, default=4, help="windows per MuQ call")
    cache.add_argument("--device", default=None)

    fit = commands.add_parser("train", help="train the projection head on the cached pairs")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--temperature", type=float, default=0.1)
    fit.add_argument("--device", default=None)

    encoding = commands.add_parser("encode", help="write fingerprints and report exact-search top-1")
    encoding.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    processed_dir = args.data_dir / "processed"

    if args.command == "cache":
        manifest = pd.read_csv(processed_dir / "manifest.csv")
        if args.limit_tracks:
            manifest = manifest.head(args.limit_tracks)
        train_ids, eval_ids = split_tracks(manifest["track_id"], args.train_frac, args.seed)
        backbone = MuQBackbone(args.device)
        cache_clean(manifest, processed_dir, backbone, args.batch_size)
        cache_clips("pairs", manifest, train_ids, args.pairs, processed_dir, backbone, True, args.seed, args.batch_size)
        cache_clips(
            "queries", manifest, eval_ids, args.queries, processed_dir, backbone, False, args.seed + 1, args.batch_size
        )
    elif args.command == "train":
        history = train(
            processed_dir,
            args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            seed=args.seed,
            device=args.device,
        )
        for epoch, (train_loss, val_loss) in enumerate(history, start=1):
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
    else:
        print(encode(processed_dir, args.model, args.device).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
