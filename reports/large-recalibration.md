# Recalibrating unknown-song thresholds on FMA Large

Reference: 104,681 tracks, 459,911 windows after removing the 1,024 unknown cohort tracks.
Query cohort: 4,096 clips from the original 2,048-song PR6 cohort. Development/test splits are unchanged.
Calibration allowed 10 false accepts among 1,024 Large development unknown clips.

## Held-out known-song results

| Retrieval | Rule | Frozen correct | Large recalibrated correct | Retained | Lost | Gained | Net change | Δ correct pp (95% song CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXACT | score only | 523/1,024 | 336/1,024 | 336 | 187 | 0 | -187 | -18.3 (-20.6, -16.0) |
| EXACT | score and gap | 691/1,024 | 336/1,024 | 328 | 363 | 8 | -355 | -34.7 (-37.8, -31.6) |
| LSH | score only | 526/1,024 | 337/1,024 | 337 | 189 | 0 | -189 | -18.5 (-20.8, -16.1) |
| LSH | score and gap | 691/1,024 | 337/1,024 | 328 | 363 | 9 | -354 | -34.6 (-37.7, -31.5) |

## Large development calibration

| Retrieval | Rule | Frozen known correct | Frozen false accepts | Recalibrated known correct | Recalibrated false accepts |
|---|---|---:|---:|---:|---:|
| EXACT | score only | 545/1,024 | 21/1,024 | 344/1,024 | 10/1,024 |
| EXACT | score and gap | 712/1,024 | 27/1,024 | 344/1,024 | 10/1,024 |
| LSH | score only | 544/1,024 | 21/1,024 | 343/1,024 | 10/1,024 |
| LSH | score and gap | 712/1,024 | 27/1,024 | 343/1,024 | 10/1,024 |

## Thresholds learned from development

| Retrieval | Rule | Score threshold | Gap threshold | Development known correct | Development unknown false accepts |
|---|---|---:|---:|---:|---:|
| EXACT | score only | 0.886316478 | — | 344/1,024 | 10/1,024 |
| EXACT | score and gap | 0.886316478 | — | 344/1,024 | 10/1,024 |
| LSH | score only | 0.886316419 | — | 343/1,024 | 10/1,024 |
| LSH | score and gap | 0.886316419 | — | 343/1,024 | 10/1,024 |

## Held-out unknown-song results

| Retrieval | Rule | Frozen false accepts | Recalibrated false accepts |
|---|---|---:|---:|
| EXACT | score only | 18/1,024 | 8/1,024 |
| EXACT | score and gap | 23/1,024 | 8/1,024 |
| LSH | score only | 18/1,024 | 8/1,024 |
| LSH | score and gap | 23/1,024 | 8/1,024 |

Calibration maximises correctly accepted known clips on development tracks subject to an empirical 1% unknown false-accept budget. The held-out test contains 1,024 known and 1,024 missing-song clips. Thresholds were selected without test labels.

For both retrievers, the best Large-development score-and-gap operating point selected no gap threshold. The recalibrated joint rule therefore coincides with score-only. Both recalibrated rules produced 8 false accepts among 1,024 held-out missing-song clips.

## Distortion-family breakdown

| Retrieval | Rule | Family | Frozen known correct | Recalibrated known correct | Frozen unknown false accepts | Recalibrated unknown false accepts |
|---|---|---|---:|---:|---:|---:|
| EXACT | score only | standard | 361 | 250 | 14 | 5 |
| EXACT | score only | tiktok | 162 | 86 | 4 | 3 |
| EXACT | score and gap | standard | 430 | 250 | 16 | 5 |
| EXACT | score and gap | tiktok | 261 | 86 | 7 | 3 |
| LSH | score only | standard | 363 | 251 | 14 | 5 |
| LSH | score only | tiktok | 163 | 86 | 4 | 3 |
| LSH | score and gap | standard | 430 | 251 | 16 | 5 |
| LSH | score and gap | tiktok | 261 | 86 | 7 | 3 |

Paired confidence intervals resample songs and keep each song's standard and TikTok-like queries together. Bootstrap point estimates were checked against the query-level rates.

## Reproduction

Run `python scripts/large_recalibration.py` from the repository root in its CuPy environment. The Large reference comes from the FMA Large build described in PR7; the query archive and ordered cohort are the saved PR6 inputs. A fresh clone also needs the pinned PR source objects:

~~~powershell
git fetch https://github.com/nickiliak/02807_computational_tools_ds.git refs/pull/1/head refs/pull/6/head refs/pull/7/head
python scripts/large_recalibration.py
~~~

The script checks the Large fingerprint checksum, PR6 query checksum and cohort IDs, and dynamically loads the LSH implementation from its pinned source commit. It writes query-level scores to the compressed CSV, along with the thresholds and song-bootstrap summaries in JSON.