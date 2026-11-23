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

## Clustering

Two uses of the general clustering algorithms from the course, both implemented by hand (k-means, DBSCAN,
Davies-Bouldin index) and tested against scikit-learn. They run on CPU in seconds on the released `small`
fingerprints.

### k-means IVF index (`src/afp/cluster.py`)

The reference windows are split into `k` clusters with spherical k-means (Lloyd's algorithm, k-means++ seeding,
centroids rescaled to unit length). A query is compared only with the windows in the `nprobe` clusters whose
centroids are closest to it. Vectors are stored sorted by cluster, so each probed cluster is one contiguous block.

```bash
uv run python -m afp.cluster --k 128 256 512 --nprobe 4 8 16 32   # writes ivf_{clusters,search}.csv
```

FMA `small`, 35,292 windows, one query at a time on a laptop CPU (numpy):

| Index | Standard top-1 | TikTok-like top-1 | Windows compared | ms per query |
|---|---:|---:|---:|---:|
| exact | 0.983 | 0.899 | 100% | 0.11 |
| k = 128, nprobe = 8 | 0.971 | 0.866 | 6.7% | 0.04 |
| k = 128, nprobe = 32 | 0.980 | 0.896 | 25% | 0.14 |
| k = 512, nprobe = 16 | 0.976 | 0.883 | 4.7% | 0.05 |

The Davies-Bouldin index keeps falling as `k` grows (4.0 at 32, 2.8 at 1024): the fingerprints have no natural
coarse cluster structure, so `k` is chosen by the speed/accuracy trade-off instead. Song-level k-means also barely
follows genre (normalised mutual information about 0.27 against `genre_top`), as expected for features trained to
tell recordings apart rather than styles.

### DBSCAN: songs that share audio (`src/afp/related.py`)

DBSCAN runs on the reference windows with cosine similarity as the neighbourhood (`eps = sqrt(2 - 2 s)` for unit
vectors). Unique music ends up as noise; a cluster holding windows of two or more tracks links those tracks.

```bash
uv run python -m afp.related --min-similarity 0.9 --min-points 2   # writes dbscan_sweep.csv, related_pairs.csv
```

At similarity 0.9 and `min_points` 2, 82.6% of windows are noise and 69 track pairs are linked, 97% of them by the
same artist: duplicate uploads, album/single versions, instrumental and vocal versions, a radio edit, a live
version and remixes, two of them across artists. Some exact-search "misses" are such versions: 5 of 34 standard
and 7 of 201 TikTok-like misses return a linked version of the right song.

## Layout

```txt
src/afp/
├── data.py        # download FMA, decode to 24 kHz mono, manifest, cut into 10 s segments
├── augment.py     # the paper's distortions (time stretch, pitch, noise, reverb, filters, echo)
├── embed.py       # MuQ backbone, 1024 -> 4096 -> 256 projection head, contrastive training
├── fingerprint.py # cache / train / encode command line
├── cluster.py     # k-means, Davies-Bouldin index, IVF index and its evaluation
└── related.py     # DBSCAN over window fingerprints to find tracks that share audio
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

### Running the GPU retrieval comparison

[Executed GPU notebook](notebooks/gpu_retrieval.ipynb) · [Medium report figure](reports/figures/gpu-operating-points-medium.pdf) · [Medium results](reports/gpu-retrieval-medium.json) · [Small results](reports/gpu-retrieval-small.json) · [Small report figure](reports/figures/gpu-retrieval-small.pdf)

The GPU notebook compares angular LSH with exhaustive cosine search on the same FP32 vectors and CUDA device. It selects a configuration using standard development tracks, then measures track identification on disjoint held-out tracks, false acceptance after removing songs, and speed for both single and batched queries. The timing starts with embeddings already in VRAM. Model inference, disk reads, host-to-device transfer and index construction are outside the retrieval timings.

On FMA medium, 110,320 windows from 24,970 searchable tracks were queried on an RTX 4090. The final evaluation used 7,458 standard and 8,992 TikTok-like queries. The two LSH settings below give different costs for reproducing exact search.

| Retrieval method | Standard top-1 | TikTok-like top-1 | Median lookup |
|---|---:|---:|---:|
| Exhaustive cosine | 98.08% | 93.54% | 0.196 ms |
| Fastest qualifying development setting (14 bits, 32 tables, 3 extra probes) | 97.25% | 89.52% | 0.052 ms |
| Highest development accuracy (10 bits, 64 tables, 3 extra probes) | 98.08% | 93.54% | 0.097 ms |

The accuracy-first comparison was added after observing the primary result. It matched exact search's correct counts on each query set and returned the same track for 16,446 of 16,450 queries. Its own interleaved exact timing was 0.207 ms. The primary index processed 1.10 million queries/s in 128-query batches versus 366,000 for exact search. At cosine threshold 0.70 it correctly accepted 83.58% of known queries and falsely accepted 5.99% of unknowns, versus 84.61% and 6.35% for exact search. These are synthetic distortions. The test includes tracks rendered unsearchable by the silence filter.

The saved settings come from the first development sweep. To select new settings using latency on a different GPU, set `AFP_RETUNE=1`. The full [GPU figure](reports/figures/gpu-retrieval-medium.pdf) also shows the development sweep, open-set outcomes and throughput. The [pipeline record](reports/medium-pipeline.json) documents hardware, source hashes, preprocessing and worker measurements. The [paired catalogue audit](reports/medium-catalogue-audit.json) checks the training/evaluation split and measures the effect of adding reference tracks with the head and query vectors held fixed.

To run it on Nikos's published fingerprints, put the three release archives in `data/processed/fingerprints`. On a fresh checkout the first command below fetches them. For the medium experiment, use the build instructions below and set `AFP_DATASET_TAG` to `medium`. The notebook saves a JSON results file and PDF/PNG figures under dataset-specific names.

```powershell
gh release download v0.2-fingerprints --repo nickiliak/02807_computational_tools_ds --pattern '*.npz' --dir data/processed/fingerprints
$env:AFP_DATASET_TAG = 'small'
uv run --isolated --no-project --python 3.12 --with-requirements notebooks/requirements-gpu.txt jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.timeout=600 notebooks/gpu_retrieval.ipynb
```

### Rebuilding the medium fingerprints on a GPU

The medium experiment starts from the original FMA audio. It fits a new head on 50,000 augmented pairs, then writes separate archives for the clean reference catalogue, 10,000 standard queries and 10,000 TikTok-like queries. All music tracks can be references. A seeded 20% of tracks supplies training pairs and the remaining 80% supplies queries. Training and test voice-overs use separate LibriSpeech speaker sets. The medium data and model live under `data/medium/`.

From the repository root, set up the locked project environment with `uv sync --locked`. The commands below use PowerShell. The FMA medium archive is 22.19 GiB; decoded waveforms, speech, model weights and intermediate features require additional storage.

```powershell
$root = 'data/medium'
$data = "$root/data"
$model = "$root/models/projection_head.pt"
$backbone = "$root/muq"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
uv run python -m afp.data --data-dir $data --subset medium
uv run python -m afp.data --data-dir $data --speech train
uv run python -m afp.data --data-dir $data --speech test
uv run hf download OpenMuQ/MuQ-large-msd-iter --revision 0562a57814f6f8bbd9fdea0a25921a2fce1a841a --local-dir $backbone
uv run python -m afp.fingerprint --data-dir $data --seed 0 cache --backbone $backbone --train-frac 0.2 --pairs 50000 --queries 10000 --voice-over --batch-size 16 --device cuda --precision bf16 --audio-cache-gb 8 --prep-workers 8
uv run python -m afp.fingerprint --data-dir $data --seed 0 cache-tiktok --backbone $backbone --train-frac 0.2 --queries 10000 --batch-size 64 --device cuda --precision bf16 --audio-cache-gb 8 --prep-workers 12
uv run python -m afp.fingerprint --data-dir $data --model $model --seed 0 train --epochs 30 --batch-size 512 --device cuda
uv run python -m afp.fingerprint --data-dir $data --model $model encode --device cuda --skip-evaluation
```

The encoder keeps MuQ in GPU memory for each cache stage and checks every batch for finite values. On a GPU with BF16 support, the option above uses BF16 for the backbone and retries non-finite batches in FP32. Audio preparation overlaps inference for augmented clips. The 8 GiB host cache holds reusable int16 source waveforms. With less available RAM, lower `--audio-cache-gb`. Each finished feature shard can be reused if a run is interrupted. Use separate data directories for experiments with different seeds, model revisions, audio, or augmentation settings.

### Checking the effect of catalogue size

After encoding FMA medium, run the following from the repository root. It checks the fingerprint shapes and norms, verifies that evaluation tracks are absent from the cached training pairs, and prints checksums and paired counts for each query set.

The comparison also needs the published small manifest. If it is missing, download it with `gh release download v0.2-fingerprints --repo nickiliak/02807_computational_tools_ds --pattern manifest.csv --dir data/processed`. Its SHA-256 is `8fe1d544d30f2a161aa4109451750621f4d6558629f4c9293f129644101ca643`.

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
