# Recording identity and five-candidate landmark verification on FMA Large

[Run the notebook](../notebooks/recording_landmarks.ipynb) to refit the rules from saved candidate scores on CPU. The [plot](recording-landmarks.pdf) and [result tables](landmark-results.json) use held-out recording groups.

## Held-out results

Each comparison uses 512 known and 512 absent clips. Both thresholded systems accepted five of 512 absent development clips at the 1% target.

| Held-out outcome | LSH cosine | Landmark votes within the five | Cosine plus landmarks |
|---|---:|---:|---:|
| Known recording first, always answer | 472 | 485 | 488 |
| Correct accepted at the 1% development target | 344 | 419 | 418 |
| Wrong title accepted for a known clip | 0 | 1 | 1 |
| Absent clip accepted | 8 | 9 | 9 |

At that operating point fusion gained 104 correct answers and lost 30. Of those gains, 94 were already first under cosine and ten needed another candidate to be promoted. The net gain was 74/512, or +14.45 percentage points (paired 95% recording-group bootstrap interval +9.96 to +19.14). Exact archive-ID scoring credited 414 of the hybrid's 418 correct accepts. The observed absent-clip rates were 1.56% for cosine and 1.76% for fusion. The latter's group-bootstrap interval was 0.59–3.32%.

Two additional frozen operating points gave identical observed false-accept counts. At the 0.25% development target cosine and fusion had three false accepts each and 185 versus 222 correct known answers. At the 2% target both accepted 12 absent clips and returned 382 versus 473 correct known answers. These policies were fitted on development before the held-out comparison.

With abstention disabled the hybrid promoted 16 true recordings without demoting a correct first choice. Of its 24 remaining errors, 22 lay outside the LSH shortlist. Twelve of these 22 came from combined speed, speech and compression, six from speed alone, three from speech alone and one from standard noise. The [per-query evidence](landmark-scores.npz) supports checking each count. A complementary candidate generator for speed changes and a real phone benchmark now address the largest remaining retrieval and validation gaps.

On 64 paired, warm RTX 4090 queries the median decoded-host-waveform to returned decision time rose from 13.87 to 16.70 ms. The median per-clip difference was +2.94 ms. The 95th percentiles were 15.14 and 18.18 ms. Method order alternated. Both paths used the same single-query group-aware LSH lookup. Its five candidate groups agreed with the evaluation shortlist on all 64 sampled clips, including when re-encoding the input live. Capture time, synthetic distortion generation, model loading and offline reference preparation are excluded. [Paired latency traces](landmark-latency.csv)

## Recording groups and fresh queries

For the 105,705 searchable track IDs, we proposed pairs with the same normalized artist and title, identical archive CRC and byte length, or similar mean neural fingerprints. For each mean vector the GPU returned six neighbours including the track itself, subject to a cosine floor of 0.90. Neither metadata nor neural similarity alone created an edge.

Of 8,446 proposed pairs, 3,446 passed the audio check. We downmixed to 2 kHz mono and searched an excerpt from one ID throughout the other. At least two non-overlapping, non-silent three-second blocks had to reach Pearson correlation 0.95 at offsets agreeing within 10 ms. Connected components produced 102,748 groups. Of those, 2,628 had multiple IDs, accounting for 5,585 archive entries. All four formerly audited, waveform-identical pairs landed in common groups. [Verified pair list](recording-alias-edges.csv.gz)

The groups describe shared audio in the supplied excerpts. A transitive group can join excerpts through an intermediate track, and a shared passage can appear in a mix. Unproposed or heavily transformed duplicates can escape detection. The exact-ID sensitivity count in the results shows how often credit depends on accepting another member of a known group.

We excluded all groups touching the Small or Medium processed manifests or the previous rejection cohort. Of the 77,962 eligible groups, seed 28071123 selected 256 groups per development/test and present/absent cell. Each contributes two ten-second synthetic queries, one per distortion family. The same search index excludes every ID in every designated absent group across both phases. The resulting searchable catalogue has 105,180 track IDs in 102,236 groups and 462,163 reference windows. The local projection training used Medium. Overlap with foundation-model pretraining is unknown.

## Acoustic verification

The common LSH search supplies five distinct recording groups. One best-window track represents each group. Given the candidate IDs and the decoded query, we extract spectral peaks at 8 kHz with a 1,024-sample Hann window and a 128-sample hop. We retain at most six prominent local maxima per 16 frames. For each candidate, a CUDA kernel votes for offsets consistent with changes of pitch and speed. A coherent run of time-frequency matches contributes independent evidence to the neural cosine. The verifier receives neither a distortion label nor a true warp factor.

We compare cosine, landmark evidence alone within that shortlist, and their weighted sum. On development data we choose the weight from 0, 0.25, 0.5, 1, 2, 4 and 8 for fusion, then optimise the winner-score and winner-minus-runner-up cutoffs jointly. We choose the operating point with the most correct accepted known clips subject to the empirical absent-clip budget. Ties in development performance are resolved deterministically. For comparisons without abstention, we choose the fusion weight by development top-one accuracy. The notebook also shows 0.25%, 2%, 5% and 10% development budgets. They are diagnostics for choosing an application policy. Each corresponding held-out point uses its frozen development choice.

## GPU reproduction

Fetch the pinned implementation once with `git fetch https://github.com/nickiliak/02807_computational_tools_ds.git refs/pull/1/head`. Start from the repository root with the original FMA Large archive and reference vectors from [the Large build](https://github.com/nickiliak/02807_computational_tools_ds/pull/7), and the Medium data files, projection checkpoint, MuQ weights and test speech. The replay notebook only needs the small committed scores and the NumPy analysis script. For a new GPU run:

1. Run `python scripts/recording_groups.py` in the pipeline's CUDA PyTorch environment to write `data/large/fresh/groups.csv` and `cohort.csv`.
2. Run `python scripts/fresh_shortlist.py` with the Medium model and CUDA retrieval environments. See its module documentation for entry points. It saves the actual query waveforms, candidate track IDs, scores and split metadata in `data/large/fresh/queries.npz`.
3. Run `python scripts/landmark_verify.py` with CuPy, NumPy, SciPy and SoundFile. Only the union of proposed candidate tracks needs reference peak maps to replay this benchmark. A deployed catalogue would prepare reference peaks for all searchable tracks before accepting traffic. Per-query verifier timings exclude offline reference preparation.
4. Run `python scripts/landmark_evaluation.py`. For paired warm one-query latency, run `python scripts/landmark_latency.py` in an environment containing both CUDA PyTorch and CuPy, then run all cells of the notebook. The per-query archive includes candidate IDs, scores from both methods, query labels and verifier timings. Raw music and speech remain local.

## Comparable work

Nikou and Giannakopoulos reported 97% recognition for ten-second low-noise mobile recordings against more than 100,000 songs. At high noise, their fifteen-second recognition rate was 56.5%. The protocol asks for the correct known song and uses different queries and noise exposures. It offers a scale and conditions for external evaluation. [Paper](https://arxiv.org/abs/2507.06070)

Sonnleitner and Widmer evaluated geometric landmark combinations on 100,000 full-length songs under changes in speed, tempo and pitch. Their construction motivates testing alignment across scale. Our five-candidate peak-voting implementation does not reproduce their quad index. [Robust Quad-Based Audio Fingerprinting](https://www.cp.jku.at/research/papers/preprint_TASLP2509248.pdf)
