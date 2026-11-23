# Where can we recover a correct title?

The listener wants the identity of a particular recording, including the option to say that it is absent. Audio fingerprinting fits that task. The open question is which representation and matching rule preserve recording identity under speech, speed changes, compression and microphone capture. Our catalogue already makes it possible to measure where recognition fails before changing the model.

## A development-only diagnostic on FMA Large

We kept the catalogue from the rejection experiment, with 104,681 searchable tracks after removing the designated unknown songs. The same 512 known development songs each contributed one standard and one TikTok-like clip. For each query, we took the highest GPU cosine for each distinct track and counted how many songs outranked the labelled recording.

| Query condition | Correct first | Correct in first five | Wrong winners with the correct track still in the first five |
|---|---:|---:|---:|
| Standard | 490/512 | 509/512 | 19 of 22 |
| TikTok-like | 460/512 | 495/512 | 35 of 52 |
| Speed + voice + compression, within TikTok-like | 91/127 | 111/127 | 20 of 36 |

The three-way distortion accounts for 36 of the 52 TikTok-like misses. The distribution across conditions is observational because different songs received different distortion labels. A perfect choice from the existing top five could recover at most 35 additional hard-family titles here. Seventeen true tracks were outside the shortlist. Rejecting uncertain winners alone cannot recover either group. The counts use strict row-order top-1 and are insensitive to optimistic versus conservative tie handling at rank five. [GPU diagnostic](../scripts/large_development_ranks.py) · [Counts by distortion](large-development-ranks.json)

To repeat this diagnostic, run `python scripts/large_development_ranks.py` from the repository root in the medium PyTorch CUDA environment with the frozen `data/large/references.npz` and `data/medium/runner-up-queries.npz` in place. [Large input checksums and build](https://github.com/nickiliak/02807_computational_tools_ds/pull/7)

## What other systems have demonstrated

| Source | Measured result | Experiment it motivates here |
|---|---|---|
| Singh et al., 2025, Table 1 | On 5,000 FMA references and random ten-second clips, MuQ reached 83.91% overall top-1 frozen and 88.18% fine-tuned. Training used 300,000 Disco-10M samples. | Compare training the projection alone with adapting the final encoder layers. Control the training pairs, optimisation budget and evaluation cohort. Their test uses different training data, catalogue size and distortions. [Paper and results](https://openreview.net/pdf?id=oaeP03QCjE) |
| Araz et al., ISMIR 2025, Tables 2 and 6 | Their final neural fingerprint reached 95.6% top-1 on ten-second synthetic queries and 84.6% on ten-second venue recordings. At two seconds the real-recording figure was 75.0%. The two evaluation datasets differ. | Measure paired direct and over-the-air captures, and vary duration. Hold out rooms, microphones and noise sources. The published synthetic-to-real difference cannot be attributed to the microphone alone. [Paper and protocols](https://arxiv.org/html/2506.22661v1) |
| Six, ISMIR 2021, Panako 2.0 | For 20-second queries sped up by 10%, the revised landmark system raised top-1 true positives from 18% to 83% relative to its 2014 implementation. | Compare a peak-based matcher to neural cosine on the same ten-second excerpts. Its agreements in time and frequency give a separate source of evidence for reranking. [Author's publication record](https://biblio.ugent.be/publication/8726851) · [Implementation](https://github.com/JorenSix/Panako) |
| Cortès-Sebastià et al., 2025 | In their TV study, separating music from speech helped spectral-peak systems especially on background music. The background-trained neural system saw little overall benefit. | Compare original and music-separated versions of the very same voice-over clips, with quiet passages and clean vocals as controls. Include the separator's runtime. [Study](https://doi.org/10.1007/s11042-025-21080-x) |
| Pex Audio Fingerprinting Benchmark Toolkit | Hard-Small supplies 100 annotated chunks assembled into 22 query files from 99 references. Its evaluator supports recording identity and aligned segments. | Use the annotations for a reproducible time-alignment test. Check for overlapping FMA track IDs before using it as an independent test. These are synthetic edits. [Toolkit, data and evaluator](https://github.com/Pexeso/audio-fingerprinting-benchmark-toolkit) |

Those top-1 rates measure whether the winner is correct when a reference is present. Our open-set outcome counts correct accepted titles **and** false accepts on absent recordings. A system can score highly on the first measure and fail the second.

For an external test on real mixtures, the BAF authors provide annotated TV audio with roughly 80% of annotated time spent on background music. Audio is available upon request. BAF tests speech and competing sounds in broadcasts. The paired FMA phone pilot answers the separate question of what a loudspeaker, room and microphone do to the same excerpt. [BAF dataset](https://zenodo.org/records/6868083) · [Paper and baseline code](https://github.com/guillemcortes/baf-dataset)

## Next experiment

1. **Test the shortlist.** Freeze the current first-stage fingerprints. For each top-five song, match short local landmarks and score whether their offsets agree along a common timeline, allowing a modest speed change. Compare unchanged cosine, landmark-only search, and the two-stage system. The earlier temporal vote used three neural windows from an 18-second clip and gained two net identifications in 384. The proposed verifier uses an independent acoustic representation on the same ten-second input. [Earlier temporal comparison](https://github.com/nickiliak/02807_computational_tools_ds/pull/3)
2. **Separate interactions.** From each fresh song and a fixed start time, generate all eight combinations of speed change, voice-over and compression. Keep speech level, speed factor and codec setting fixed where active. Repeat by severity on development tracks. Pair by song to estimate whether the combined damage exceeds the sum of individual changes. Audit tied IDs and duplicate recordings separately.
3. **Choose the rule before the final test.** Select configurations on development songs. Hold out fresh songs, speakers and physical capture conditions for confirmation. Report top-five recall, correct accepted knowns, unknown false accepts, wrong known titles, and timing from decoded audio through the final decision. Keep this run's held-out cohort as a historical comparison. It has already informed the next experiment.
