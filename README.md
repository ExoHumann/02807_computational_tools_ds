# Neural audio fingerprinting

Project for DTU course 02807 Computational Tools for Data Science. Deadline: 4 December 2026.

This project builds a song identification system, similar to Shazam or Google voice that can recognise a song from a short and possibly distorted TikTok like audio clip. We follow [Robust Neural Audio Fingerprinting using Music Foundation Models](https://arxiv.org/abs/2511.05399), we use a pretrained foundation model (MuQ, the paper's best-performing backbone) to turn short segments of audio into fingerprints i.e: compact vector representations designed to stay similar under distortions such as noise, compression, time-acceleration, background noise etc. To go through this library of fingerprints the idea is to use Locality-Sensitive Hashing (LSH) and try to measure the trade-off between retrieval accuracy and speed against exhaustive search. We finally build a graph of songs that can share matching parts and apply graph clustering (Louvain) to uncover non-obvious connections between tracks, such as remixes covers or samples. The system is evaluated on clips with synthetic distortions, including a held-out set of TikTok-like distortions (sped-up playback, voice-over, compression) that the model never trains on.

## Data

We use the [Free Music Archive](https://github.com/mdeff/fma) (FMA) `small` subset: 8,000 clips of 30 s (mp3) plus
metadata (Defferrard et al., ISMIR 2017). 7,994 decode cleanly; 6 corrupt mp3s are skipped. Audio is under each
artist's own licence and meant for research, so it is downloaded rather than redistributed. It needs about 7.7 GB for
the zip plus about 11.5 GB of processed audio. The `medium` subset (25,000 clips, about 22 GB zip plus 36 GB
processed) works the same way with `--subset medium`.

```bash
uv run python -m afp.data --subset small            # download, then process everything
uv run python -m afp.data --subset small --limit 50 # quick test on the first 50 tracks
```

The fingerprints, the trained head and `manifest.csv` for `small` are published as the GitHub release
`v0.2-fingerprints` (head trained with voice-over; `v0.1-fingerprints` has the first head), so the audio is only
needed to fingerprint new clips or re-run the pipeline.

The download is resumable and checked against the published SHA-1. Each clip is decoded straight from the zip to
24 kHz mono int16 and saved as `data/processed/audio/<NNN>/<track_id>.npy`. Re-running skips finished tracks.
`data/processed/manifest.csv` lists every processed track (id, path, duration, title, artist, genre) and
`skipped.csv` lists tracks that could not be decoded or are shorter than 10 s.

## Fingerprints

Three resumable steps turn the processed audio into fingerprints (`src/afp/fingerprint.py`):

```bash
uv run python -m afp.fingerprint cache    # MuQ features for every window, training pairs and evaluation queries
uv run python -m afp.fingerprint train    # contrastive training of the 1024 -> 4096 -> 256 projection head
uv run python -m afp.fingerprint encode   # write the fingerprints and print exact-search top-1 per distortion
```

Each 30 s clip is cut into 10 s windows with a 5 s hop (35,326 windows for the 7,994 tracks of `small`). MuQ is frozen;
only the projection head is trained, on pairs of a clean window and a distorted copy of it (`src/afp/augment.py`).
The tracks are split 20/80: the 20% only provide training pairs, and the evaluation queries come from the other 80%,
so the head is never scored on tracks it learned from. Use `cache --limit-tracks 200 --pairs 500 --queries 100` for a
quick trial run.

Output: `data/processed/fingerprints/references.npz` (`vectors` as `float32 [N, 256]` with unit norm, plus
`track_id`, `segment_idx`, `start_s`, `rms`) and `queries.npz` (the same plus the `augmentation` category). The head is
saved to `models/projection_head.pt`. `encode` drops near-silent reference windows (`--min-rms`, default 0.001; 34 in
`small`), because every silent window gets the same fingerprint and would match silent windows of other tracks.

`afp.fingerprint.search(references, track_id, queries, threshold)` returns the best track per query, or `-1` when the
best cosine similarity is below `threshold` ("song not in the database"). `encode` also reports, for a range of
thresholds, how many queries are still found and how many queries of songs missing from the references (half of the
query tracks, removed for this test) are wrongly matched. Around 0.7 is a reasonable default.

Speed: on CPU MuQ takes about 1.6 s per window (measured here), so `cache` with the defaults (about 153,000 windows)
would take roughly three days. It needs a GPU: `pyproject.toml` installs the CUDA 11.8 build of torch. On a DTU HPC
V100 it took about 0.07 s per window, and the full run on FMA `small` (`hpc/setup_small.sh`, then `hpc/run_small.sh`)
took about 1.5 hours.

Distortions (`src/afp/augment.py`): time stretch (speed changes, pitch kept), pitch shift (up to 5 semitones, speed
kept), both combined, coloured noise, reverb, reverb plus noise, band-, high- and low-pass filters, and echo. They are
synthetic stand-ins chosen to need no extra downloads, and they are used both for training and for the standard
queries.

TikTok-like test queries: three further distortions build held-out queries: sped-up playback (1.1x to 1.5x, pitch
rises too), voice-over (real speech from LibriSpeech, CC BY 4.0, mixed at -5 to +10 dB music-to-speech; the level is
saved per query as `voice_over_db`) and MP3 compression (about 27 to 40 kbit/s), each alone and all three combined.
They still do not cover real phone microphones or TikTok's own processing.

Voice-over can also be added to training (`cache --voice-over`). Training speech comes from LibriSpeech
`train-clean-100` and test speech from `dev-clean`; LibriSpeech splits have disjoint speakers, so no test voice is
heard during training. Sped-up playback, compression and the combined category stay unseen.

```bash
uv run python -m afp.data --speech train          # 600 speech chunks from 100 train-clean-100 speakers
uv run python -m afp.data --speech test           # 240 speech chunks from the 40 dev-clean speakers
uv run python -m afp.fingerprint cache --voice-over --pairs 10000 --queries 2000
uv run python -m afp.fingerprint cache-tiktok     # MuQ features of 2,000 held-out TikTok-like queries
uv run python -m afp.fingerprint encode           # also scores queries_tiktok when its features exist
```

Differences from the paper: noise and reverb are synthetic instead of MUSAN (real noise, speech and music recordings)
and Aachen (real room impulse responses), there is no Encodec compression, and the open MuQ checkpoint was trained on
the Million Song Dataset, so it may not reproduce the paper's numbers. MuQ weights are CC-BY-NC 4.0.

Evaluation so far (exact search, FMA `small`, queries from tracks the head never trained on, 2,000 per set):

| Query set | Head v1 | Head v2 (with voice-over) |
|---|---|---|
| standard distortions | 0.981 | 0.983 |
| TikTok-like: compression | 0.986 | 0.989 |
| TikTok-like: sped up | 0.962 | 0.951 |
| TikTok-like: voice-over | 0.393 | 0.973 |
| TikTok-like: all three combined | 0.234 | 0.645 |
| TikTok-like: overall | 0.651 | 0.899 |

With a 0.7 threshold, head v2 finds 0.925 of standard and 0.668 of TikTok-like queries whose song is in the
references, and wrongly accepts 0.050 and 0.034 of queries whose song is not. Real-world recordings are not covered.

## Layout

```txt
src/afp/
├── data.py        # download FMA, decode to 24 kHz mono, manifest, cut into 10 s segments
├── augment.py     # the paper's distortions (time stretch, pitch, noise, reverb, filters, echo)
├── embed.py       # MuQ backbone, 1024 -> 4096 -> 256 projection head, contrastive training
└── fingerprint.py # cache / train / encode command line
tests/
data/{raw,processed}/   # git-ignored contents
notebooks/
```

## LSH retrieval benchmark

[Open the executed notebook](notebooks/lsh_retrieval.ipynb) for the track-disjoint parameter sweep, exact and LSH results by distortion, and CPU timing comparison. The [report figure (PDF)](reports/figures/lsh-retrieval.pdf) also has a [PNG version](reports/figures/lsh-retrieval.png).

From the repository root, open the notebook with the lightweight CPU environment and select **Run All Cells**:

```bash
uv run --isolated --no-project --python 3.12 --with-requirements notebooks/requirements-lsh.txt jupyter lab notebooks/lsh_retrieval.ipynb
```

To regenerate the executed notebook and both figures from a terminal:

```bash
uv run --isolated --no-project --python 3.12 --with-requirements notebooks/requirements-lsh.txt jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.timeout=600 notebooks/lsh_retrieval.ipynb
```

The notebook verifies the three v0.2 fingerprint archives in `data/processed/fingerprints/` against the published digests and downloads any missing archive. These vectors are the full input to this experiment. The index lives in `src/afp/lsh.py`, and the original exact baseline in `src/afp/retrieval.py` remains available as `afp.fingerprint.search`.

### Checking the effect of catalogue size

After encoding FMA medium, run the following from the repository root. It checks the fingerprint shapes and norms, verifies that evaluation tracks are absent from the cached training pairs, and prints checksums and paired counts for each query set.

```bash
uv run --isolated --no-project --python 3.12 --with-requirements notebooks/requirements-gpu.txt python scripts/compare_catalogues.py data/medium/data/processed
```

The comparison uses the new medium head for both searches. It restricts the smaller index to tracks in the v0.2 small manifest and evaluates exactly the same queries against both catalogues. This isolates the effect of adding candidate songs. The original small release uses its own head and query draws.

## Development

Uses [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv sync
uv run pytest tests/
uv run invoke --list      # test, data, fmt
```
