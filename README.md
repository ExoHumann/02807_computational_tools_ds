# Neural audio fingerprinting

Project for DTU course 02807 Computational Tools for Data Science. Deadline: 4 December 2026.

This project builds a song identification system, similar to Shazam or Google voice that can recognise a song from a short and possibly distorted TikTok like audio clip. We follow [Robust Neural Audio Fingerprinting using Music Foundation Models](https://arxiv.org/abs/2511.05399), we use a pretrained foundation model (MERT, or MuQ which is the paper's best backbone) to turn short segments of audio into fingerprints i.e: compact vector representations designed to stay similar under distortions such as noise, compression, time-acceleration, background noise etc. To go through this library of fingerprints the idea is to use Locality-Sensitive Hashing (LSH) and try to measure the trade-off between retrieval accuracy and speed against exhaustive search. We finally build a graph of songs that can share matching parts and apply graph clustering (Louvain) to uncover non-obvious connections between tracks, such as remixes covers or samples. The system will be evaluated on real TikTok clips.

## Layout

```txt
src/afp/
├── data.py        # list tracks, decode to 24 kHz mono, cut into 10 s segments
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
uv run invoke --list      # test, fmt
```
