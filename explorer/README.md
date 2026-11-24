# Signal Index

The Vercel app serves the interactive results and a self-contained 24-song spectral-landmark matcher. It keeps working while the training PC is offline. The local service uses the released 24,970-song Medium index, MuQ and the trained projection head on CUDA, four angular buckets per table, 64 tables and a distinct-track score gap. Bucket selection runs on the CPU and cosine reranking runs on CUDA. The displayed timing for each live result is separate from the archived GPU benchmark.

## Listening

Capture ten seconds or select a browser-decodable 10–30 second clip up to 8 MiB. The browser downmixes and resamples its first ten seconds and sends 480,000 bytes of mono PCM16. Both APIs validate the content type and length. No audio is saved by the handlers. The track list and the licensed Jay Laurier clip let a visitor test the matcher with known audio.

The hosted index uses 24 FMA Medium tracks labelled CC0 in the FMA metadata. Its extractor follows PR 9 (8 kHz spectral peaks); its CPU matcher votes for one temporal offset at original speed. Its heuristic cutoffs are 24 peaks, alignment score 0.32 and gap 0.12. With these settings fixed, digital crops beginning at sample 137,203 produced 24 correct accepts in 24 known songs and zero accepts in 48 other songs. These are clean digital controls. There are no labelled phone captures in this result. The [individual outcomes](hosted-check.json) and the catalogue builder expose the sampling and decision details.

## Explore the evidence

The search chart compares the exact, LSH and IVF runs in PR 13 for 7,458 standard and 8,992 TikTok-like clips per catalogue. Latency starts after fingerprint extraction. The browser inspector exposes the 2,048 held-out PR 6 decisions, including withheld source, distortion, predicted ID, runner-up, cosine and gap. The genre table uses PR 2's separate test cohort. The evidence builder checks both acceptance totals against their archived sources. Catalogue and query identities come from the released Medium manifest.

## Run

For Vercel, select the project's `explorer` root. It serves `public/` and the two functions in `api/`. Deployment does not need the neural model. An optional HTTPS worker can be configured with server-side `SIGNAL_GPU_URL` (full recognition URL) and `SIGNAL_GPU_TOKEN`; the function falls back to landmarks if the worker is unavailable. Cold workers exceeding the seven-second upstream timeout also use the fallback.

For local neural recognition use the repository GPU environment, the Medium release directory and head, and the pinned OpenMuQ checkpoint. Set `MUQ_MODEL_DIR` to the local model directory if it is outside the original cache location. Run `./explorer/start.ps1` and open http://127.0.0.1:8327. Stop with `./explorer/stop.ps1`. The web service uses loopback. No tunnel is required for a local test.

Rebuild the public inputs from the repository root with `python explorer/tools/build_evidence.py` (requires the PR 13 and 14 commits), `python explorer/tools/build_landmarks.py` (requires FMA Medium audio and metadata), then `python explorer/tools/check_hosted.py`. The hosted dependencies are listed in requirements.txt. The builders also use the project's pandas environment.

[FMA metadata and dataset](https://github.com/mdeff/fma) · [CC0](https://creativecommons.org/publicdomain/zero/1.0/) · [Report PR 13](https://github.com/nickiliak/02807_computational_tools_ds/pull/13) · [Appendix PR 14](https://github.com/nickiliak/02807_computational_tools_ds/pull/14)

