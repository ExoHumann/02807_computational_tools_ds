"""Audit the development unknowns accepted by the recalibrated exact rule.

Run in the medium CUDA PyTorch environment after large_recalibration.py.
"""

import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly

RATE = 4000
WIDTH = 5 * RATE


def excerpt(archive, track):
    name = f"fma_large/{track // 1000:03d}/{track:06d}.mp3"
    raw, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32", always_2d=True)
    mono = resample_poly(raw.mean(axis=1), RATE, rate)
    return torch.tensor(mono, dtype=torch.float64, device="cuda")


def local_correlations(query, reference):
    """Peak Pearson correlation of each non-silent source block at any reference offset."""
    size = 1 << (len(reference) + WIDTH - 2).bit_length()
    spectrum = torch.fft.rfft(reference, size)
    prefix = torch.nn.functional.pad(reference.cumsum(0), (1, 0))
    square = torch.nn.functional.pad(reference.square().cumsum(0), (1, 0))
    energy = square[WIDTH:] - square[:-WIDTH] - (prefix[WIDTH:] - prefix[:-WIDTH]).square() / WIDTH
    maxima = []
    for window in query.unfold(0, WIDTH, WIDTH):
        centred = window - window.mean()
        if centred.square().mean() < 1e-6:
            continue
        convolution = torch.fft.irfft(spectrum * torch.fft.rfft(centred.flip(0), size), size)
        numerator = convolution[WIDTH - 1 : len(reference)]
        denom = (energy.clamp_min(1e-12) * centred.square().sum()).sqrt()
        corr = torch.where(energy / WIDTH >= 1e-6, numerator / denom, 0)
        maxima.append(float(corr.max().cpu()))
    return maxima


def main():
    reports = Path("reports")
    result = json.loads((reports / "large-recalibration.json").read_text(encoding="utf8"))
    threshold = result["methods"]["exact"]["comparison"]["score_and_gap"]["large_recalibrated"]
    with gzip.open(reports / "large-recalibration-scores.csv.gz", "rt", encoding="utf8") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["method"] == "exact"
            and row["phase"] == "development"
            and row["known"] == "False"
            and float(row["top_score"]) >= threshold["score_threshold"]
            and (threshold["gap_threshold"] is None or float(row["gap"]) >= threshold["gap_threshold"])
        ]
    assert len(rows) == threshold["development_unknown_false_accepts"]
    wanted = {int(row[field]) for row in rows for field in ("track_id", "top_track")}
    with zipfile.ZipFile("data/medium/data/raw/fma_metadata.zip") as meta:
        name = next(name for name in meta.namelist() if name.endswith("/tracks.csv"))
        with io.TextIOWrapper(meta.open(name), encoding="utf8") as source:
            records = csv.reader(source)
            header = list(zip(next(records), next(records), strict=True))
            ai, ti = header.index(("artist", "name")), header.index(("track", "title"))
            next(records)
            labels = {int(row[0]): [row[ai], row[ti]] for row in records if int(row[0]) in wanted}
    findings = []
    with zipfile.ZipFile("data/large/fma_large.zip") as archive:
        for row in rows:
            source, winner = int(row["track_id"]), int(row["top_track"])
            source_audio, winning_audio = excerpt(archive, source), excerpt(archive, winner)
            correlations = local_correlations(source_audio, winning_audio)
            findings.append(
                {
                    "source_id": source,
                    "winning_id": winner,
                    "identical_full_mono_4khz_waveform": torch.equal(source_audio, winning_audio),
                    "source_artist_title": labels[source],
                    "winning_artist_title": labels[winner],
                    "top_score": float(row["top_score"]),
                    "gap": float(row["gap"]),
                    "nonoverlapping_source_windows": len(correlations),
                    "windows_at_or_above_0.95": sum(c >= 0.95 for c in correlations),
                    "max_five_second_correlation": max(correlations, default=None),
                }
            )
    (reports / "large-development-impostors.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf8")
    print("Pairs, same artist, same artist and title, and pairs with a 5s correlation >=0.95:")
    print(
        len(findings),
        sum(x["source_artist_title"][0] == x["winning_artist_title"][0] for x in findings),
        sum(x["source_artist_title"] == x["winning_artist_title"] for x in findings),
        sum(x["windows_at_or_above_0.95"] > 0 for x in findings),
    )


if __name__ == "__main__":
    main()
