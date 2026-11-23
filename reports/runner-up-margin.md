# Does the gap to the runner-up help reject unknown songs?

## Fresh-track test

We compared a maximum cosine threshold with a rule that also requires a sufficient score gap between the best two distinct tracks. Development tracks select thresholds to maximise correct acceptance of known queries with no more than 1% empirical false acceptance among unknown queries. The held-out tracks evaluate both frozen rules. The exact GPU run scored 4,096 synthetic queries against 105,767 remaining reference windows on an NVIDIA RTX 4090.

| Rule | Known correct accepted | Known accepted incorrectly | Unknown false accepted | Standard correct | TikTok-like correct |
|---|---:|---:|---:|---:|---:|
| Maximum score | 538/1024 (52.5%) | 2 | 4/1024 (0.39%) | 369/512 (72.1%) | 169/512 (33.0%) |
| Score plus distinct-song margin | 731/1024 (71.4%) | 0 | 11/1024 (1.07%) | 454/512 (88.7%) | 277/512 (54.1%) |

On the same held-out known queries, the margin rule changed correct acceptance by +18.85 percentage points (paired track-bootstrap 95% interval +16.31 to +21.39). Unknown false acceptance changed by +0.68 points (95% paired interval +0.10 to +1.37).

The margin rule accepted 11/1,024 missing-song queries, one above its 10/1,024 development budget. The score-only rule accepted 4/1,024. The gap rule correctly accepted 731/1,024 known queries versus 538/1,024 for score-only. It accepted no incorrect known titles in this test, compared with two for score-only. This is a measured trade-off, not a demonstrated 1% test FAR.

Track-bootstrap 95% intervals, thresholds, and family false accepts are in runner-up-margin.json. Per-query scores for the best windows from two distinct track IDs are in runner-up-margin-scores.csv.gz. The ordered cohort IDs in runner-up-margin-cohort-tracks.csv preserve the held-out split for reproduction.

The exact GPU scorer compares each query to all available reference windows, reduces window scores to one maximum per track, and chooses two distinct track IDs. The query cohort excludes training tracks, both earlier query archives, and the earlier unknown-song cohort.

The margin increases correct acceptance by substantially more than it increases false acceptance in this sample. Its observed test false-acceptance rate still exceeds the development target. Synthetic distortions cannot establish phone-in-room performance.

## Reproduction

Run from the repository root in PowerShell with the processed medium data and the pinned model files:

~~~powershell
$python = 'data/medium/.venv/Scripts/python.exe'
$backbone = (Resolve-Path 'data/medium/hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots/0562a57814f6f8bb9dfea0a25921a2fce1a841a').Path
& $python scripts/runner_up_study.py generate --backbone $backbone
& $python scripts/runner_up_study.py score
& $python scripts/runner_up_study.py analyse
~~~

The ordered track IDs and earlier study IDs are in runner-up-margin-cohort-tracks.csv. The JSON stores checksums for the reference fingerprints, generated query embeddings, projection head and cohort IDs. Running analyse alone rebuilds the JSON and this report from the saved GPU scores.
