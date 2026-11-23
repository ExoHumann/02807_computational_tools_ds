# Independent acoustic candidate search on FMA Large

Of the 22 clips from 21 recording groups that LSH missed at five, independent acoustic search returned the source for 13. After fusion, 13 ranked in the final five, 13 ranked first with abstention disabled, and 9 received a correct accepted answer at the 1% development target. Widening LSH to ten retrieved one of the same 22. Acoustic search also found that clip.

We reused the fixed ten-second queries, projection model, duplicate groups and held-out split. Each phase contains 512 known and 512 absent clips. Both candidate generators searched 105,180 reference tracks. LSH selected five distinct recording groups. The acoustic search used pairs of spectral peaks to vote for a recording and time offset at 29 playback rates from 0.80 to 1.50. It checked the alignment of its leading 64 tracks before proposing five groups. On development, this recovered five of 16 LSH misses, compared with four for peak pairs alone and zero for invariant triples. The union retained the original five and averaged 9.60 groups per held-out clip. The ten-group LSH control used the same verifier. Each method selected its fusion weight on development songs. At the 1% target it also selected the winner-score and runner-up-gap thresholds under a budget of five absent accepts.

| Method | In pool | Final five | First | Correct accepted | Wrong known | False absent | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSH 5 | 490 | 490 | 488 | 418 | 1 | 9 | 15.7 | 16.7 |
| LSH 10 | 491 | 491 | 489 | 418 | 1 | 9 | 15.9 | 16.7 |
| LSH + acoustic | 503 | 503 | 500 | 427 | 1 | 10 | 18.3 | 19.5 |

At the 1% target the dual system gained 9 correct accepted answers and lost 0 relative to LSH 5. The paired change was +1.76 percentage points (95% recording-group bootstrap interval +0.59 to +3.12). Using the original LSH 5 weight and thresholds unchanged on the expanded pool gave 427 correct accepts and 10 false accepts.

On 64 paired, warm RTX 4090 queries, parallel search took 18.35 ms median versus 15.74 ms for LSH 5 and 15.90 ms for LSH 10. Running the dual branches serially took 20.22 ms. Each timer starts with the decoded waveform and ends at the returned decision. It includes encoding, landmark extraction, lookup, verification and abstention. Capture, file decoding and offline index construction are excluded.

At the separately fitted 2% development target, the combined system accepted 488 correct titles and 11 absent clips. LSH 5 accepted 473 correct titles and 12 absent clips. At the 0.25% target both methods accepted three absent clips, with correct titles increasing from 222 to 226.

**Original misses by distortion.** speed_up: 6/6, speed_up+voice_over+compression: 4/12, voice_over: 2/3, noise: 1/1.

The original 22 are a retrospective diagnostic. Recognition is scored by verified recording group. The acoustic ranking alone placed the source in its five nominations on 395/512 known clips. The held-out false-accept counts are observations after calibration on separate songs.

[CPU replay and figure](../notebooks/acoustic_search.ipynb) · [Report figure](acoustic-search.pdf) · [22 individual clips](acoustic-original-misses.csv) · [Thresholds and operating points](acoustic-results.json) · [Timing samples](acoustic-latency.csv)

## GPU candidate generation

Candidate peaks use the same 8 kHz spectrogram, 1,024-sample window, 128-sample hop and local maximum filter as PR 9. The searchable pair index holds 907,107,488 postings (3.38 GiB), plus its offsets and a host posting-count array. Each key stores two quantized log frequencies and a 9–42-frame lag. For each hypothesized speed, the search probes rare keys across the clip and sums postings at nearby offsets. On the 64 best tracks an independent landmark verifier checks coherent pitch and time transformations. The neural model, original five LSH candidates, and synthetic queries are frozen. The comparison chose acoustic geometry using only known development clips. An earlier coarse frequency/lag encoding with a 768-posting cutoff identified 16/512 development sources in its acoustic shortlist and recovered zero of 16 development LSH misses. Narrower bins and less aggressive pruning improved that result. [Development selection](acoustic-selection.json).

## Reproduction

Run all notebook cells from this checkout to replay the tables and regenerate the figure from the included per-candidate evidence. The notebook needs NumPy, pandas, Matplotlib and Jupyter.

For the full GPU experiment use the FMA Large archive, reference embeddings, Medium projection head, cached MuQ model and [PR 9’s duplicate-aware query preparation](https://github.com/nickiliak/02807_computational_tools_ds/pull/9). Use its frozen groups.csv, cohort.csv, queries.npz, queries-embedded.npz, pinned_lsh implementation and candidate-landmarks.npz. Fetch the pinned Git objects with `git fetch https://github.com/nickiliak/02807_computational_tools_ds.git refs/pull/9/head refs/pull/1/head`. From the repository root with CUDA PyTorch, CuPy, MuQ, SoundFile, SciPy and NumPy installed, run:

```shell
python scripts/acoustic_catalogue.py --batch 8 --workers 16
python scripts/acoustic_search.py
```

The first command checkpoints peak shards below data/large/fresh/acoustic. The second builds the searchable index, scores all 2,048 clips, fits development operating points and times live queries. Offline acoustic preparation uses CPU workers to decode audio and the GPU to extract spectral peaks. All raw audio stays local.

**Live-timing replay check.** Out of 64 clips: LSH 5 64 identical candidate sets and 64 identical returned decisions, LSH 10 63 identical candidate sets and 64 identical returned decisions, LSH + acoustic 64 identical candidate sets and 64 identical returned decisions.
