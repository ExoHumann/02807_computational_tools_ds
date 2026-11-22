# Rejecting unknown songs on fresh FMA medium tracks

At a target of 1% false acceptance on development data, accuracy-first LSH falsely accepted **8 of 1,024** missing-song test queries (0.78%, track-bootstrap 95% interval 0.29–1.46%). It correctly identified and accepted **515 of 1,024** known queries (50.29%, 47.46–53.03%). Exact cosine had the same totals. The target costs recognition, particularly for TikTok-like queries: 29.88% correct known acceptance compared with 70.70% on standard distortions. A test false-acceptance rate above 1% remains consistent with this sample.

The head and LSH settings were frozen before generating queries. We drew 2,048 searchable evaluation tracks absent from the previous standard and TikTok query archives, excluding the seeded training split and checking the actual training-pair shards. Half the songs went into development and half into test. Each track supplied one standard and one TikTok-like query. We removed every window from 1,024 predetermined songs across both splits to simulate missing tracks in one shared catalogue. The outcome unit is a query; confidence intervals resample tracks. New query features use FP32 inference and the same frozen medium head as the catalogue. This study uses synthetic distortions.

[Executed notebook](../notebooks/unknown_songs.ipynb) · [Figure](figures/unknown-songs.pdf) · [Per-query GPU scores](unknown-song-scores.csv.gz) · [Results](unknown-songs.json)

## Run the analysis

From the repository root:

```sh
uv run --isolated --no-project --python 3.12 --with pandas --with matplotlib --with numpy --with ipykernel --with nbconvert jupyter nbconvert --execute --inplace --to notebook notebooks/unknown_songs.ipynb
```

## Regenerate the GPU scores

Use the [medium pipeline and checkpoint configuration](https://github.com/ExoHumann/02807_computational_tools_ds/blob/5e9427ee5f83cd32fdd9b5a1c8f40396c3772989/README.md#rebuilding-the-medium-fingerprints-on-a-gpu) to obtain processed audio, the medium head, prior query files, training shards and speech bank. The default head checksum identifies the measured run. For an independently rebuilt catalogue pass its matching head and checksum via `--model` and `--expected-head-sha256`. Run the generator in the project's Torch environment and the scorer in a CuPy environment. The following PowerShell commands are for a CUDA 12 GPU retrieval environment.

```powershell
$commit = '5e9427ee5f83cd32fdd9b5a1c8f40396c3772989'
$source = "https://raw.githubusercontent.com/ExoHumann/02807_computational_tools_ds/$commit/src/afp"
New-Item -Force -ItemType Directory data/review-baseline | Out-Null
foreach ($file in 'gpu_retrieval.py','gpu_lsh.cu') {
    Invoke-WebRequest "$source/$file" -OutFile "data/review-baseline/$file"
}
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
uv run python scripts/unknown_song_queries.py --backbone data/medium/muq
$env:CUPY_TF32 = '0'
uv run --isolated --no-project --python 3.12 --with 'cupy-cuda12x[ctk]>=14.2,<15' --with pandas python scripts/score_unknown_songs.py
```
