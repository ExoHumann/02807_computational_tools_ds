# Neural audio fingerprinting

Project for DTU course 02807 Computational Tools for Data Science. Deadline: 4 December 2026.

This project builds a song identification system, similar to Shazam or Google voice that can recognise a song from a short and possibly distorted TikTok like audio clip. We follow [Robust Neural Audio Fingerprinting using Music Foundation Models](https://arxiv.org/abs/2511.05399), we use a pretrained foundation model (MERT, or MuQ which is the paper's best backbone) to turn short segments of audio into fingerprints i.e: compact vector representations designed to stay similar under distortions such as noise, compression, time-acceleration, background noise etc. To go through this library of fingerprints the idea is to use Locality-Sensitive Hashing (LSH) and try to measure the trade-off between retrieval accuracy and speed against exhaustive search. We finally build a graph of songs that can share matching parts and apply graph clustering (Louvain) to uncover non-obvious connections between tracks, such as remixes covers or samples. The system will be evaluated on real TikTok clips.

## Data

We use the [Free Music Archive](https://github.com/mdeff/fma) (FMA) `medium` subset: 25,000 clips of 30 s (mp3) plus
metadata (Defferrard et al., ISMIR 2017). Audio is under each artist's own licence and meant for research, so it is
downloaded rather than redistributed. It needs about 22 GB for the zip plus about 36 GB of processed audio.

```bash
uv run python -m afp.data --subset medium            # download, then process everything
uv run python -m afp.data --subset medium --limit 50 # quick test on the first 50 tracks
```

The download is resumable and checked against the published SHA-1. Each clip is decoded straight from the zip to
24 kHz mono int16 and saved as `data/processed/audio/<NNN>/<track_id>.npy`. Re-running skips finished tracks.
`data/processed/manifest.csv` lists every processed track (id, path, duration, title, artist, genre) and
`skipped.csv` lists tracks that could not be decoded or are shorter than 10 s.

## Layout

```txt
src/afp/
├── data.py        # download FMA, decode to 24 kHz mono, manifest, cut into 10 s segments
├── augment.py     # the paper's distortions (not implemented yet)
└── embed.py       # backbone + 1024 -> 4096 -> 256 projection head
tests/
data/{raw,processed}/   # git-ignored contents
notebooks/
```

## Development

Uses [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv sync
uv run pytest tests/
uv run invoke --list      # test, data, fmt
```
