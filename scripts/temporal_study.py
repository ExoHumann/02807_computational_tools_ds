"""Build synthetic controls and a labelled playback set, or evaluate supplied phone recordings."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from afp.augment import random_augment
from afp.data import SAMPLE_RATE, load_audio, load_processed, load_speech_bank
from afp.embed import MuQBackbone
from afp.fingerprint import load_head, split_tracks
from afp.temporal import match_windows

SEED = 17953
QUERY_S = 18
WINDOW_S = 10
STARTS = (0, 4, 8)
TOP_K = 12
CONDITIONS = {
    "direct": (),
    "room_noise": ("reverb", "noise"),
    "speed_voice_mp3": ("speed_up", "voice_over", "compression"),
}
PHONE = ("near_quiet", "near_speech", "far_quiet")


def select_songs(processed: Path) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Reserve separate song sets for digital controls and later physical captures."""
    manifest = pd.read_csv(processed / "manifest.csv")
    _, eligible = split_tracks(manifest.track_id, 0.2, 0)
    for name in ("queries", "queries_tiktok"):
        with np.load(processed / f"fingerprints/{name}.npz") as query:
            eligible = np.setdiff1d(eligible, query["track_id"])
    with np.load(processed / "fingerprints/references.npz") as refs:
        eligible = np.intersect1d(eligible, refs["track_id"])
    chosen = np.random.default_rng(SEED).choice(eligible, 140, replace=False)
    return manifest.set_index("track_id").path, chosen[:128], chosen[128:]


def prepare_pack(processed: Path, pack: Path) -> None:
    """Make 12 local playback WAVs and a recording manifest with one slot per condition."""
    paths, _, tracks = select_songs(processed)
    (pack / "playback").mkdir(parents=True, exist_ok=True)
    (pack / "recordings").mkdir(exist_ok=True)
    rows = []
    for number, track in enumerate(tracks, start=1):
        code = f"song-{number:02d}"
        wave = load_processed(processed / paths[track])
        first = (len(wave) - 20 * SAMPLE_RATE) // 2
        sf.write(pack / "playback" / f"{code}.wav", wave[first : first + 20 * SAMPLE_RATE], SAMPLE_RATE)
        rows.extend(
            {"track_id": int(track), "condition": cond, "file": f"recordings/{code}-{cond}.wav"} for cond in PHONE
        )
    with (pack / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "condition", "file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(tracks)} playback files and {len(rows)} recording slots in {pack}")
    (pack / "README.txt").write_text(
        "Play each WAV on one speaker at a fixed volume. Record it with the same phone "
        "near the speaker (0.5 m, quiet), near with speech, and farther away (3 m, quiet). "
        "Trim each WAV to the playback with at least 18 seconds of audio. Save it under "
        "the path listed in manifest.csv. Note actual distances, phone, speaker and room. "
        "See reports/temporal-study.md in the project for scoring.\n",
        encoding="utf-8",
    )


def centered(wave: np.ndarray) -> np.ndarray:
    """Use the same central 18 seconds for both matching methods."""
    length = QUERY_S * SAMPLE_RATE
    if len(wave) < length:
        raise ValueError(f"An 18-second clip is required; got {len(wave) / SAMPLE_RATE:.2f} seconds")
    first = (len(wave) - length) // 2
    return wave[first : first + length].copy()


class Catalogue:
    """FP32 MuQ and projection head with resident GPU reference fingerprints."""

    def __init__(self, processed: Path, head: Path, checkpoint: str) -> None:
        self.encoder = MuQBackbone("cuda", repo=checkpoint)
        self.head = load_head(head).cuda().eval()
        with np.load(processed / "fingerprints/references.npz") as ref:
            self.vectors = torch.from_numpy(ref["vectors"]).cuda()
            self.tracks = torch.from_numpy(ref["track_id"]).cuda()
            self.times = torch.from_numpy(ref["start_s"]).cuda()
        self.starts = torch.tensor(STARTS, device="cuda", dtype=torch.float32)

    @torch.inference_mode()
    def candidates(self, clips: list[np.ndarray]) -> tuple:
        """Encode the same three windows and retrieve ranked neighbours on CUDA."""
        windows = np.stack([x[s * SAMPLE_RATE : (s + WINDOW_S) * SAMPLE_RATE] for x in clips for s in STARTS])
        features = [self.encoder.encode(windows[i : i + 16]) for i in range(0, len(windows), 16)]
        vector = self.head(torch.from_numpy(np.concatenate(features)).cuda())
        score, row = (vector @ self.vectors.T).topk(TOP_K)
        shape = (len(clips), len(STARTS), TOP_K)
        return score.reshape(shape), self.tracks[row].reshape(shape), self.times[row].reshape(shape)

    def predict(self, clips: list[np.ndarray]) -> tuple:
        return match_windows(*self.candidates(clips), self.starts)


def elapsed(catalogue: Catalogue, clips: list[np.ndarray]) -> dict:
    """Warm, alternate method order and time full clip-to-prediction at batch size one."""
    catalogue.predict([clips[0]])
    times = {"strongest": [], "temporal": []}
    for i, clip in enumerate(clips):
        methods = ("strongest", "temporal") if i % 2 == 0 else ("temporal", "strongest")
        for method in methods:
            torch.cuda.synchronize()
            clock = perf_counter()
            scores, tracks, ref_times = catalogue.candidates([clip])
            if method == "strongest":
                tracks.flatten()[scores.flatten().argmax()].item()
            else:
                match_windows(scores, tracks, ref_times, catalogue.starts)[1].item()
            torch.cuda.synchronize()
            times[method].append(1000 * (perf_counter() - clock))
    return {
        k: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "n": len(v)}
        for k, v in times.items()
    }


def evaluate(args: argparse.Namespace) -> None:
    """Run matched comparisons; real-phone mode reads only captures actually present."""
    paths, tracks, _ = select_songs(args.processed)
    if args.mode == "phone":
        manifest = pd.read_csv(args.pack / "manifest.csv")
        entries = [row for row in manifest.itertuples() if (args.pack / row.file).is_file()]
        print(f"Found {len(entries)} of {len(manifest)} expected phone recordings")
        if not entries:
            raise SystemExit("No phone recordings supplied; no real-world accuracy was computed")

        def read(entry):
            return int(entry.track_id), entry.condition, centered(load_audio(args.pack / entry.file))
    else:
        entries = [(int(t), name) for t in tracks for name in CONDITIONS]
        speech = load_speech_bank(args.processed / "speech_test.npy")

        def read(entry):
            track, cond = entry
            rng = np.random.default_rng([SEED, track, list(CONDITIONS).index(cond)])
            wave = load_processed(args.processed / paths[track])
            if CONDITIONS[cond]:
                wave, _, _ = random_augment(wave, rng, SAMPLE_RATE, (CONDITIONS[cond],), speech)
            return track, cond, centered(wave)

    catalogue = Catalogue(args.processed, args.model, args.backbone)
    rows, sample_clips = [], []
    sample_ids = set(np.random.default_rng(SEED).choice(len(entries), min(24, len(entries)), replace=False))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, len(entries), 16):
            samples = list(pool.map(read, entries[first : first + 16]))
            clips = [x[2] for x in samples]
            strongest, temporal, support = [a.cpu().tolist() for a in catalogue.predict(clips)]
            for i, (track, condition, _) in enumerate(samples):
                rows.append(
                    {
                        "track_id": track,
                        "condition": condition,
                        "strongest": strongest[i],
                        "temporal": temporal[i],
                        "support_windows": support[i],
                    }
                )
                if first + i in sample_ids:
                    sample_clips.append(clips[i])
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / f"temporal-{args.mode}.csv", index=False)
    (args.output / f"temporal-{args.mode}-timing.json").write_text(
        json.dumps(elapsed(catalogue, sample_clips), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["controls", "pack", "phone"])
    parser.add_argument("--processed", type=Path, default=Path("data/medium/data/processed"))
    parser.add_argument("--model", type=Path, default=Path("data/medium/models/projection_head.pt"))
    parser.add_argument("--backbone", default="data/medium/muq")
    parser.add_argument("--pack", type=Path, default=Path("data/medium/phone-study"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "pack":
        prepare_pack(args.processed, args.pack)
    else:
        evaluate(args)
