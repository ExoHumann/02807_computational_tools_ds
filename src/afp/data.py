"""Data pipeline: download FMA, decode it to 24 kHz mono int16 ``.npy`` files and build a manifest.

Run ``uv run python -m afp.data --subset medium`` to download into ``data/raw`` and process into ``data/processed``.
"""

import argparse
import hashlib
import http.client
import io
import os
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 24_000
SEGMENT_S = 10.0
HOP_S = 5.0

FMA_URL = "https://os.unil.cloud.switch.ch/fma"
FMA_SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
    "fma_medium.zip": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
}
CHUNK_BYTES = 1 << 20
MAX_RETRIES = 5

_zip: zipfile.ZipFile | None = None


def download(url: str, dest: Path, sha1: str | None = None) -> None:
    """Download ``url`` to ``dest``, resuming a partial download and verifying the SHA-1 checksum.

    A ``<dest>.verified`` marker is written once the file is complete and checked, so later calls return at once.

    Args:
        url: File to download. The server must support HTTP range requests to resume.
        dest: Destination path. Parent directories are created.
        sha1: Expected SHA-1 hex digest, or ``None`` to skip verification.

    Raises:
        RuntimeError: If the downloaded file does not match ``sha1`` or the download keeps failing.
    """
    dest = Path(dest)
    marker = dest.with_name(f"{dest.name}.verified")
    if dest.exists() and marker.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part")

    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])
    if part.exists() and part.stat().st_size > total:
        part.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        done = part.stat().st_size if part.exists() else 0
        if done == total:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    done = 0
                with open(part, "ab" if done else "wb") as out, tqdm(
                    total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
                ) as progress:
                    while chunk := response.read(CHUNK_BYTES):
                        out.write(chunk)
                        progress.update(len(chunk))
        except (OSError, http.client.HTTPException) as error:
            print(f"download interrupted ({error}), retry {attempt}/{MAX_RETRIES}")
    if (part.stat().st_size if part.exists() else 0) != total:
        raise RuntimeError(f"could not download {url}")

    part.replace(dest)
    if sha1 is not None:
        with open(dest, "rb") as f:
            digest = hashlib.file_digest(f, "sha1").hexdigest()
        if digest != sha1:
            dest.unlink()
            raise RuntimeError(f"checksum mismatch for {dest.name}: got {digest}, expected {sha1}")
    marker.touch()


def load_audio(source: Path | BinaryIO, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to a mono ``float32`` waveform at ``sample_rate``.

    Args:
        source: Path or binary file object holding any format ``soundfile`` can decode (mp3, wav, flac, ...).
        sample_rate: Target sample rate in Hz. 24 kHz is what MuQ and MERT expect.

    Returns:
        A 1D ``float32`` array.
    """
    data, rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if rate != sample_rate:
        waveform = torchaudio.functional.resample(torch.from_numpy(waveform), rate, sample_rate).numpy()
    return np.ascontiguousarray(waveform, dtype=np.float32)


def segment(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segment_s: float = SEGMENT_S,
    hop_s: float = HOP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a waveform into fixed-length, possibly overlapping windows.

    Trailing audio shorter than one window is dropped, and a waveform shorter than one window yields no segments.

    Args:
        waveform: 1D waveform.
        sample_rate: Sample rate of ``waveform`` in Hz.
        segment_s: Window length in seconds.
        hop_s: Distance between window starts in seconds.

    Returns:
        A tuple ``(segments, start_s)``: ``segments`` has shape ``[n, window_samples]`` and ``start_s`` holds the
        start time of each window in seconds.

    Raises:
        ValueError: If ``waveform`` is not 1D or the window or hop is shorter than one sample.
    """
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1D, got shape {waveform.shape}")
    window = round(segment_s * sample_rate)
    hop = round(hop_s * sample_rate)
    if window < 1 or hop < 1:
        raise ValueError("segment_s and hop_s must each cover at least one sample")
    if len(waveform) < window:
        return np.empty((0, window), dtype=waveform.dtype), np.empty(0, dtype=np.float64)
    segments = np.lib.stride_tricks.sliding_window_view(waveform, window)[::hop].copy()
    start_s = np.arange(len(segments)) * hop / sample_rate
    return segments, start_s


def track_path(processed_dir: Path, track_id: int) -> Path:
    """Return where the processed waveform of a track is stored: ``audio/<first 3 digits>/<id>.npy``."""
    name = f"{track_id:06d}"
    return Path(processed_dir) / "audio" / name[:3] / f"{name}.npy"


def _init_worker(zip_path: str) -> None:
    """Open the audio zip once per worker process."""
    global _zip
    torch.set_num_threads(1)
    _zip = zipfile.ZipFile(zip_path)


def _process_member(job: tuple[str, str]) -> tuple[int, str | None]:
    """Decode one zip member and save it as int16. Returns ``(track_id, reason)`` with ``reason`` set if skipped."""
    member, out_path = job
    track_id = int(Path(member).stem)
    try:
        with _zip.open(member) as f:
            waveform = load_audio(io.BytesIO(f.read()))
    except Exception as error:
        return track_id, f"decode failed: {error}"
    if len(waveform) < round(SEGMENT_S * SAMPLE_RATE):
        return track_id, f"shorter than {SEGMENT_S:g} s"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f"{out.name}.tmp")
    with open(temporary, "wb") as f:
        np.save(f, np.round(np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16))
    temporary.replace(out)
    return track_id, None


def process_zip(
    zip_path: Path, processed_dir: Path, limit: int | None = None, workers: int | None = None
) -> pd.DataFrame:
    """Decode every mp3 in an FMA zip to 24 kHz mono int16 ``.npy`` files, in parallel and without extracting.

    Tracks whose output already exists are skipped, so an interrupted run can simply be repeated. Tracks that fail
    to decode or are shorter than one segment are not written.

    Args:
        zip_path: FMA audio zip (e.g. ``fma_medium.zip``).
        processed_dir: Output directory; waveforms go to ``audio/<NNN>/<track_id>.npy`` below it.
        limit: Process only the first ``limit`` tracks (sorted by name), or all if ``None``.
        workers: Number of worker processes. Defaults to the CPU count minus one.

    Returns:
        A table with columns ``track_id`` and ``reason`` for tracks skipped in this run.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(name for name in zf.namelist() if name.endswith(".mp3"))
    if limit:
        members = members[:limit]
    jobs = []
    for member in members:
        out = track_path(processed_dir, int(Path(member).stem))
        if not out.exists():
            jobs.append((member, str(out)))

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        _init_worker(str(zip_path))
        results = list(tqdm(map(_process_member, jobs), total=len(jobs), desc="processing"))
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(zip_path),)) as pool:
            results = list(tqdm(pool.map(_process_member, jobs, chunksize=8), total=len(jobs), desc="processing"))
    skipped = [(track_id, reason) for track_id, reason in results if reason is not None]
    return pd.DataFrame(skipped, columns=["track_id", "reason"])


def load_track_metadata(metadata_zip: Path) -> pd.DataFrame:
    """Read title, artist and top-level genre of every track from ``fma_metadata.zip``.

    Args:
        metadata_zip: Path to ``fma_metadata.zip``.

    Returns:
        A table with columns ``track_id``, ``title``, ``artist`` and ``genre_top``.
    """
    with zipfile.ZipFile(metadata_zip) as zf:
        name = next(n for n in zf.namelist() if Path(n).name == "tracks.csv")
        with zf.open(name) as f:
            tracks = pd.read_csv(f, index_col=0, header=[0, 1])
    track_id = pd.to_numeric(tracks.index, errors="coerce")
    table = pd.DataFrame(
        {
            "track_id": track_id,
            "title": tracks[("track", "title")].to_numpy(),
            "artist": tracks[("artist", "name")].to_numpy(),
            "genre_top": tracks[("track", "genre_top")].to_numpy(),
        }
    )
    return table.dropna(subset=["track_id"]).astype({"track_id": int})


def build_manifest(processed_dir: Path, metadata_zip: Path | None = None) -> pd.DataFrame:
    """Describe every processed track, derived from the files on disk so it is always consistent with them.

    Args:
        processed_dir: Directory that :func:`process_zip` wrote to.
        metadata_zip: ``fma_metadata.zip`` to join title, artist and genre from, or ``None`` to skip.

    Returns:
        A table with ``track_id``, ``path`` (relative to ``processed_dir``), ``n_samples`` and ``duration_s``, plus
        ``title``, ``artist`` and ``genre_top`` when ``metadata_zip`` is given.
    """
    rows = []
    for path in sorted((Path(processed_dir) / "audio").rglob("*.npy")):
        n_samples = np.load(path, mmap_mode="r").shape[0]
        rows.append(
            {
                "track_id": int(path.stem),
                "path": path.relative_to(processed_dir).as_posix(),
                "n_samples": n_samples,
                "duration_s": n_samples / SAMPLE_RATE,
            }
        )
    manifest = pd.DataFrame(rows, columns=["track_id", "path", "n_samples", "duration_s"])
    if metadata_zip is not None:
        manifest = manifest.merge(load_track_metadata(metadata_zip), on="track_id", how="left")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Command line entry point: download FMA, process it and write ``manifest.csv`` and ``skipped.csv``."""
    parser = argparse.ArgumentParser(description="Download FMA and process it to 24 kHz mono int16 .npy files.")
    parser.add_argument("--subset", choices=["small", "medium"], default="medium")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N tracks")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count - 1)")
    parser.add_argument("--skip-download", action="store_true", help="use the zips already in data/raw")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    raw_dir, processed_dir = args.data_dir / "raw", args.data_dir / "processed"
    audio_name = f"fma_{args.subset}.zip"
    if not args.skip_download:
        for name in ("fma_metadata.zip", audio_name):
            download(f"{FMA_URL}/{name}", raw_dir / name, FMA_SHA1[name])

    skipped = process_zip(raw_dir / audio_name, processed_dir, args.limit, args.workers)
    skipped.to_csv(processed_dir / "skipped.csv", index=False)
    manifest = build_manifest(processed_dir, raw_dir / "fma_metadata.zip")
    manifest.to_csv(processed_dir / "manifest.csv", index=False)
    print(f"{len(manifest)} tracks in manifest, {len(skipped)} skipped this run")


if __name__ == "__main__":
    main()
