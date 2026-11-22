# Time-offset voting for distorted music

On 128 songs under three conditions, time-offset voting identified 363 of 384 clips, compared with 361 for the strongest match. Seven clips improved and five regressed. The paired change of +0.52 percentage points had a 95% track-bootstrap interval of −1.56 to +2.34 points. The speed-up/voice-over/MP3 condition accounted for the clearest change, from 108 to 112 correct out of 128. The other conditions each lost one. These results justify a physical-capture test while leaving the overall benefit unresolved. With decoded audio ready, median clip-to-track time on an RTX 4090 was 35.50 ms for temporal voting and 35.37 ms for the strongest match.

The baseline picks the best-scoring window anywhere in an 18-second clip. The alternative asks whether several windows identify the same song at consistent reference times. For example, query windows at 0, 4 and 8 seconds matching a song at 5, 9 and 13 seconds all imply a five-second offset. A wrong track can have the single highest cosine and still lose a vote across windows.

We fix three windows at 0, 4 and 8 seconds, 12 neighbours per window and a three-second offset tolerance. We take the strongest positive score per query window for each candidate track and offset, then sum. Each method receives the same GPU exhaustive-search results. Large speed changes can stretch relative offsets, and several neighbouring query windows share audio. Track-level paired analysis accounts for repeated conditions on a song. The underlying offset idea is described in [Wang, ISMIR 2003](https://ismir2003.ismir.net/presentations/Wang.pdf).

[Executed analysis](../notebooks/temporal_voting.ipynb) · [Report figure](figures/temporal-voting.pdf) · [Paired predictions](temporal-controls.csv) · [Clip-to-track timings](temporal-controls-timing.json)

## Reproduce

The analysis notebook runs from the committed control predictions and timings. Run from the repository root:

```sh
uv run --isolated --no-project --python 3.12 --with numpy --with pandas --with matplotlib --with ipykernel --with nbconvert jupyter nbconvert --execute --inplace --to notebook notebooks/temporal_voting.ipynb
```

For the full CUDA experiment, first build the [medium data, projection head, speech bank and pinned MuQ checkpoint](https://github.com/ExoHumann/02807_computational_tools_ds/blob/5e9427ee5f83cd32fdd9b5a1c8f40396c3772989/README.md#rebuilding-the-medium-fingerprints-on-a-gpu). From the repo root, set OMP, OpenBLAS and MKL thread counts to 1 before running concurrent audio preparation. These commands use the project Torch environment:

```sh
uv run python scripts/temporal_study.py controls --backbone data/medium/muq
uv run python scripts/temporal_study.py pack
```

The controls use 128 seeded songs from the head's evaluation split, excluding songs from the previous standard and TikTok-like query sets. Each supplies direct digital audio, synthetic reverberation plus noise, and sped-up audio mixed with speech and MP3-compressed. A separate set of 12 tracks is reserved for physical recordings. Control timings start with decoded 18-second waveforms and include windowing, GPU feature extraction, projection, exact search and selection of the predicted track. Disk I/O and offline distortion generation are excluded.

## Physical listening test

FMA medium supplies the original excerpts. Generating the pack writes local playback WAVs and a labelled `manifest.csv` under `data/medium/phone-study`. Each WAV runs for 20 seconds. Use the same speaker, phone and playback volume for the three conditions below.

| Condition | Capture |
|---|---|
| near_quiet | Place the phone about 0.5 m from the speaker in a quiet room. |
| near_speech | Keep the near setup and speak over the playback. |
| far_quiet | Move the phone about 3 m away in the same room. |

Start recording before playback, stop just after the excerpt, and trim lead-in and trailing silence. Save WAVs in the manifest's recording paths; convert other formats to WAV if needed. Record the phone model, speaker, distances and room so the results can be interpreted. The scorer uses the central 18 seconds. It reports how many planned files were present and writes a separate file of predictions for actual captures:

```sh
uv run python scripts/temporal_study.py phone --backbone data/medium/muq --output data/medium/phone-study/results
```

The shipped controls are digital. Real-phone performance is pending physical capture.
