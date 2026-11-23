"""Create a fresh, duplicate-aware LSH shortlist cohort for landmark reranking.

Run the query preparation with the cached Medium CUDA environment:
    data/medium/.venv/Scripts/python.exe scripts/fresh_shortlist.py

The script writes two 10-second clips per fresh song, embeds them with the frozen
MuQ projection, and delegates only the CuPy LSH shortlist stage to the repository
retrieval environment. A separate mode is used internally by that subprocess.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "data" / "large" / "fresh"
ARCHIVE = ROOT / "data" / "large" / "fma_large.zip"
REFERENCE = ROOT / "data" / "large" / "references.npz"
GROUPS = FRESH / "groups.csv"
COHORT = FRESH / "cohort.csv"
INTERMEDIATE = FRESH / "queries-embedded.npz"
OUTPUT = FRESH / "queries.npz"
LSH_COMMIT = "5e9427ee5f83cd32fdd9b5a1c8f40396c3772989"
LSH_SETTINGS = {"bits": 10, "tables": 64, "probes": 3, "seed": 2807}
SAMPLE_RATE = 24_000
WINDOW_SAMPLES = 10 * SAMPLE_RATE


def load_cohort() -> tuple[dict[int, int], list[dict[str, object]], dict[int, str]]:
    """Read and validate the duplicate map and held-out group split."""
    group_by_track: dict[int, int] = {}
    with GROUPS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            track_id, group_id = int(row["track_id"]), int(row["group_id"])
            if track_id in group_by_track:
                raise ValueError(f"duplicate group-map row for track {track_id}")
            group_by_track[track_id] = group_id

    cohort: list[dict[str, object]] = []
    phase_by_group: dict[int, str] = {}
    status_by_group: dict[int, bool] = {}
    with COHORT.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            track_id, group_id = int(row["track_id"]), int(row["group_id"])
            phase, known = row["phase"], bool(int(row["known"]))
            if phase not in {"development", "test"} or track_id not in group_by_track:
                raise ValueError(f"invalid cohort row {row}")
            if group_by_track[track_id] != group_id:
                raise ValueError(f"track {track_id} does not match the group map")
            if group_id in phase_by_group and phase_by_group[group_id] != phase:
                raise ValueError(f"recording group {group_id} leaks across development and test")
            if group_id in status_by_group and status_by_group[group_id] != known:
                raise ValueError(f"recording group {group_id} is both present and absent")
            phase_by_group[group_id] = phase
            status_by_group[group_id] = known
            cohort.append({"track_id": track_id, "group_id": group_id, "phase": phase, "known": known})

    cells = Counter((str(row["phase"]), bool(row["known"])) for row in cohort)
    if len(cohort) != 1024 or set(cells.values()) != {256}:
        raise ValueError(f"expected 256 distinct groups per phase/status cell, got {cells}")
    if len(phase_by_group) != len(cohort):
        raise ValueError("the cohort must contain distinct recording groups")
    return group_by_track, cohort, {int(k): str(v) for k, v in phase_by_group.items()}


def distortion_plan(cohort: list[dict[str, object]], test_categories: tuple[tuple[str, ...], ...]):
    """Stratify TikTok-like edits evenly within each of the four cohort cells."""
    plan: dict[tuple[int, str], tuple[str, ...]] = {}
    for cell_number, cell in enumerate(
        (("development", False), ("development", True), ("test", False), ("test", True))
    ):
        rows = [row for row in cohort if (row["phase"], row["known"]) == cell]
        tracks = np.asarray([int(row["track_id"]) for row in rows], dtype=np.int32)
        permutation = np.random.default_rng(2807 + cell_number).permutation(tracks)
        if len(permutation) % len(test_categories):
            raise ValueError("each cohort cell must divide evenly across TikTok categories")
        per_category = len(permutation) // len(test_categories)
        for position, track_id in enumerate(permutation):
            plan[int(track_id), cell[0]] = test_categories[position // per_category]
    return plan


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_queries() -> dict[str, object]:
    """Decode fresh FMA excerpts, apply paired distortions, then encode on CUDA."""
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "medium" / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT / "src"))

    import torch

    from afp.augment import CATEGORIES, TEST_CATEGORIES, random_augment
    from afp.data import load_audio, load_speech_bank
    from afp.embed import MuQBackbone
    from afp.fingerprint import load_head

    if not torch.cuda.is_available():
        raise RuntimeError("the fresh query encoder must run on the CUDA GPU")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    _, cohort, _ = load_cohort()
    tiktok_plan = distortion_plan(cohort, TEST_CATEGORIES)
    row_by_track = {int(row["track_id"]): row for row in cohort}
    track_ids = np.asarray([int(row["track_id"]) for row in cohort], dtype=np.int32)
    n_queries = len(track_ids) * 2
    standard_audio = np.memmap(
        FRESH / "query-audio-standard.f32", mode="w+", dtype=np.float32,
        shape=(len(track_ids), WINDOW_SAMPLES),
    )
    tiktok_audio = np.memmap(
        FRESH / "query-audio-tiktok.f32", mode="w+", dtype=np.float32,
        shape=(len(track_ids), WINDOW_SAMPLES),
    )
    family = np.tile(np.asarray(["standard", "tiktok"], dtype="U8"), len(track_ids))
    phase = np.empty(n_queries, dtype="U11")
    known = np.empty(n_queries, dtype=np.bool_)
    group_ids = np.empty(n_queries, dtype=np.int32)
    source_tracks = np.repeat(track_ids, 2)
    distortion = np.empty(n_queries, dtype="U48")
    speech_path = ROOT / "data" / "medium" / "data" / "processed" / "speech_test.npy"
    speech = load_speech_bank(speech_path) if any("voice_over" in names for names in TEST_CATEGORIES) else None

    decode_start = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for song_index, track_id in enumerate(track_ids):
            member = f"fma_large/{track_id // 1000:03d}/{track_id:06d}.mp3"
            source = load_audio(__import__("io").BytesIO(archive.read(member)), SAMPLE_RATE)
            if len(source) < WINDOW_SAMPLES:
                raise ValueError(f"track {track_id} is shorter than the query window")

            metadata = row_by_track[int(track_id)]
            phase_name = str(metadata["phase"])
            known_flag = bool(metadata["known"])
            group_id = int(metadata["group_id"])
            # One common musical passage is used for both distortion families.
            clip_rng = np.random.default_rng([2807, int(track_id), 24])
            maximum_start = max(len(source) - int(1.5 * WINDOW_SAMPLES), 0)
            original_start = int(clip_rng.integers(maximum_start + 1))

            standard_rng = np.random.default_rng([2807, int(track_id), 0])
            standard_category = CATEGORIES[int(standard_rng.integers(len(CATEGORIES)))]
            standard, standard_rate, standard_label = random_augment(
                source, standard_rng, SAMPLE_RATE, (standard_category,)
            )
            standard_start = round(original_start / standard_rate)
            if standard_start + WINDOW_SAMPLES > len(standard):
                raise ValueError(f"standard distortion shortened the source window for track {track_id}")

            tiktok_names = tiktok_plan[int(track_id), phase_name]
            tiktok_rng = np.random.default_rng([2807, int(track_id), 1])
            tiktok, tiktok_rate, tiktok_label = random_augment(
                source, tiktok_rng, SAMPLE_RATE, (tiktok_names,), speech
            )
            tiktok_start = round(original_start / tiktok_rate)
            if tiktok_start + WINDOW_SAMPLES > len(tiktok):
                raise ValueError(f"TikTok distortion shortened the source window for track {track_id}")

            index = 2 * song_index
            standard_audio[song_index] = standard[standard_start : standard_start + WINDOW_SAMPLES]
            tiktok_audio[song_index] = tiktok[tiktok_start : tiktok_start + WINDOW_SAMPLES]
            phase[index : index + 2] = phase_name
            known[index : index + 2] = known_flag
            group_ids[index : index + 2] = group_id
            distortion[index] = standard_label
            distortion[index + 1] = tiktok_label
            if (song_index + 1) % 64 == 0:
                standard_audio.flush()
                tiktok_audio.flush()
                print(f"decoded and distorted {song_index + 1}/{len(track_ids)} songs", flush=True)
    standard_audio.flush()
    tiktok_audio.flush()
    decode_seconds = time.perf_counter() - decode_start

    print("Loading cached MuQ backbone on the RTX 4090", flush=True)
    backbone = MuQBackbone(device=str(device))
    head = load_head(ROOT / "data" / "medium" / "models" / "projection_head.pt").to(device).eval()
    vectors = np.empty((n_queries, 256), dtype=np.float32)

    # Warm the model with the first batch, then time the actual cohort with GPU syncs.
    batch_size = 8
    with torch.inference_mode():
        warm_input = np.stack(
            [standard_audio[0], tiktok_audio[0], standard_audio[1], tiktok_audio[1],
             standard_audio[2], tiktok_audio[2], standard_audio[3], tiktok_audio[3]]
        )
        warm_audio = torch.from_numpy(warm_input).to(device)
        warm_features = backbone.model(warm_audio, output_hidden_states=True).last_hidden_state.mean(dim=1).float()
        warm_vectors = head(warm_features)
        torch.cuda.synchronize(device)
        del warm_audio, warm_features, warm_vectors

        encode_start = time.perf_counter()
        for first in range(0, n_queries, batch_size):
            stop = min(first + batch_size, n_queries)
            indices = np.arange(first, stop, dtype=np.int32)
            songs = indices // 2
            batch_input = np.empty((len(indices), WINDOW_SAMPLES), dtype=np.float32)
            for row, (query_index, song_index) in enumerate(zip(indices, songs, strict=True)):
                source_map = standard_audio if query_index % 2 == 0 else tiktok_audio
                batch_input[row] = source_map[song_index]
            batch = torch.from_numpy(batch_input).to(device)
            features = backbone.model(batch, output_hidden_states=True).last_hidden_state.mean(dim=1).float()
            projected = head(features)
            vectors[first:stop] = projected.cpu().numpy()
            del batch, features, projected
            if stop % 256 == 0:
                print(f"MuQ encoded {stop}/{n_queries} clips", flush=True)
        torch.cuda.synchronize(device)
        encode_seconds = time.perf_counter() - encode_start

    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
        raise ValueError("MuQ projection produced invalid or non-unit query fingerprints")

    embeddings: dict[str, object] = {
        "vectors": vectors,
        "track_id": source_tracks,
        "group_id": group_ids,
        "phase": phase,
        "known": known,
        "family": family,
        "distortion": distortion,
        "sample_rate": np.asarray(SAMPLE_RATE, dtype=np.int32),
    }
    np.savez(INTERMEDIATE, **embeddings)
    standard_audio.flush()
    tiktok_audio.flush()
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "queries": n_queries,
        "songs": len(track_ids),
        "group_disjoint": True,
        "frontend_seconds": decode_seconds,
        "muq_projection_seconds": encode_seconds,
        "muq_seconds_per_query": encode_seconds / n_queries,
        "frontend_seconds_per_query": decode_seconds / n_queries,
        "cuda_device": torch.cuda.get_device_name(device),
        "encoder_batch_size": batch_size,
        "reference_sha256": digest(REFERENCE),
        "groups_sha256": digest(GROUPS),
        "cohort_sha256": digest(COHORT),
        "standard_categories": ["+".join(category) for category in CATEGORIES],
        "tiktok_categories_per_cell": {
            f"{phase_name}:{'known' if known_flag else 'absent'}": dict(
                Counter(
                    str(distortion[2 * index + 1])
                    for index, row in enumerate(cohort)
                    if row["phase"] == phase_name and row["known"] == known_flag
                )
            )
            for phase_name, known_flag in (
                ("development", False), ("development", True), ("test", False), ("test", True)
            )
        },
    }
    (FRESH / "query-encoding.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return embeddings


def run_search_subprocess() -> dict[str, object]:
    """Search the embeddings in the existing CuPy runtime, then package all arrays."""
    FRESH.mkdir(parents=True, exist_ok=True)
    root_python = ROOT / ".venv" / "Scripts" / "python.exe"
    command = [str(root_python), str(Path(__file__).resolve()), "--search-only"]
    environment = os.environ.copy()
    environment["CUPY_CACHE_DIR"] = str(FRESH / "cupy-cache")
    Path(environment["CUPY_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return package_outputs()


def package_outputs() -> dict[str, object]:
    """Combine paired audio, query labels, embeddings and the saved LSH shortlist."""
    with np.load(INTERMEDIATE) as encoded, np.load(FRESH / "shortlist.npz") as shortlist:
        fields = {name: encoded[name] for name in encoded.files}
        fields.update({name: shortlist[name] for name in shortlist.files})
        sys.path.insert(0, str(ROOT / "src"))
        from afp.augment import TEST_CATEGORIES

        for phase_name in ("development", "test"):
            for known_flag in (False, True):
                cell = (
                    (fields["phase"] == phase_name)
                    & (fields["known"] == known_flag)
                    & (fields["family"] == "tiktok")
                )
                counts = Counter(map(str, fields["distortion"][cell]))
                expected = Counter({"+".join(category): 64 for category in TEST_CATEGORIES})
                if counts != expected:
                    raise ValueError(f"TikTok condition balance differs in {phase_name}/{known_flag}: {counts}")
        standard_audio = np.memmap(
            FRESH / "query-audio-standard.f32", mode="r", dtype=np.float32,
            shape=(1024, WINDOW_SAMPLES),
        )
        tiktok_audio = np.memmap(
            FRESH / "query-audio-tiktok.f32", mode="r", dtype=np.float32,
            shape=(1024, WINDOW_SAMPLES),
        )
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            audio_header = {
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                "fortran_order": False,
                "shape": (2048, WINDOW_SAMPLES),
            }
            with archive.open("audio.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array_header_1_0(member, audio_header)
                for song_index in range(1024):
                    member.write(standard_audio[song_index].tobytes(order="C"))
                    member.write(tiktok_audio[song_index].tobytes(order="C"))
            for name, array in fields.items():
                buffer = io.BytesIO()
                np.save(buffer, array)
                archive.writestr(f"{name}.npy", buffer.getvalue())
        with zipfile.ZipFile(OUTPUT) as archive, archive.open("audio.npy") as member:
            version = np.lib.format.read_magic(member)
            if version != (1, 0):
                raise ValueError(f"unsupported audio NPY header version {version}")
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
            if shape != (2048, WINDOW_SAMPLES) or fortran_order or dtype != np.dtype(np.float32):
                raise ValueError("packaged audio array has an unexpected NPY header")
            if member.read(20) != standard_audio[0, :5].tobytes(order="C"):
                raise ValueError("packaged first query differs from the standard audio memmap")
    timing_path = FRESH / "query-encoding.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["tiktok_categories_per_cell"] = {}
    timing["standard_distortions_per_cell"] = {}
    for phase_name in ("development", "test"):
        for known_flag in (False, True):
            suffix = "known" if known_flag else "absent"
            cell = (fields["phase"] == phase_name) & (fields["known"] == known_flag)
            tiktok = cell & (fields["family"] == "tiktok")
            standard = cell & (fields["family"] == "standard")
            key = f"{phase_name}:{suffix}"
            timing["tiktok_categories_per_cell"][key] = dict(
                Counter(map(str, fields["distortion"][tiktok]))
            )
            timing["standard_distortions_per_cell"][key] = dict(
                Counter(map(str, fields["distortion"][standard]))
            )
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    timing.update(json.loads((FRESH / "shortlist-timing.json").read_text(encoding="utf-8")))
    timing.update({"artifact": str(OUTPUT.relative_to(ROOT)), "artifact_bytes": OUTPUT.stat().st_size})
    (FRESH / "fresh-shortlist-timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return timing


def search_only() -> None:
    """Build the pinned LSH index and extract five distinct recording groups per query."""
    os.environ.setdefault("CUPY_CACHE_DIR", str(FRESH / "cupy-cache"))
    Path(os.environ["CUPY_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import cupy as cp

    module_dir = FRESH / "pinned_lsh"
    module_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("gpu_retrieval.py", "gpu_lsh.cu"):
        content = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{LSH_COMMIT}:src/afp/{filename}"]
        )
        (module_dir / filename).write_bytes(content)
    spec = importlib.util.spec_from_file_location("fresh_shortlist_lsh", module_dir / "gpu_retrieval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned LSH implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    group_by_track, cohort, _ = load_cohort()
    absent_groups = np.asarray(sorted({int(row["group_id"]) for row in cohort if not row["known"]}), dtype=np.int32)
    with np.load(REFERENCE) as archive:
        reference_vectors = archive["vectors"].astype(np.float32, copy=False)
        reference_tracks = archive["track_id"].astype(np.int32, copy=False)
    track_groups = np.full(int(reference_tracks.max()) + 1, -1, dtype=np.int32)
    for track_id, group_id in group_by_track.items():
        if track_id < len(track_groups):
            track_groups[track_id] = group_id
    reference_groups = track_groups[reference_tracks]
    if np.any(reference_groups < 0):
        missing = np.unique(reference_tracks[reference_groups < 0])[:10]
        raise ValueError(f"reference tracks missing group assignments, examples {missing}")
    keep = ~np.isin(reference_groups, absent_groups)
    reference_vectors = np.ascontiguousarray(reference_vectors[keep], dtype=np.float32)
    reference_tracks = reference_tracks[keep]
    reference_groups = reference_groups[keep]
    if np.isin(reference_groups, absent_groups).any():
        raise ValueError("an absent recording group remains in the reference catalogue")

    group_ids = np.unique(reference_groups)
    group_lookup = np.full(len(track_groups), -1, dtype=np.int32)
    group_lookup[group_ids] = np.arange(len(group_ids), dtype=np.int32)
    reference_group_codes = group_lookup[reference_groups]
    if np.any(reference_group_codes < 0):
        raise ValueError("could not encode reference recording groups")

    cp.cuda.Device(0).use()
    references = cp.asarray(reference_vectors)
    tracks = cp.asarray(reference_tracks)
    group_codes = cp.asarray(reference_group_codes)
    build_start = time.perf_counter()
    index = module.GPUAngularLSH(
        references,
        tracks,
        bits=LSH_SETTINGS["bits"],
        tables=LSH_SETTINGS["tables"],
        seed=LSH_SETTINGS["seed"],
    )
    cp.cuda.Stream.null.synchronize()
    index_build_seconds = time.perf_counter() - build_start
    bucket_sizes = index.offsets[:, 1:] - index.offsets[:, :-1]
    maximum_bucket = int(cp.max(bucket_sizes).item())
    n_groups = len(group_ids)
    top5_tracks: list[np.ndarray] = []
    top5_groups: list[np.ndarray] = []
    top5_scores: list[np.ndarray] = []
    top1_tracks: list[np.ndarray] = []
    top1_scores: list[np.ndarray] = []
    search_seconds = 0.0
    shortlist_seconds = 0.0
    timed_queries = 0
    batch_size = 8
    chunk_size = 65_536
    tables = cp.arange(LSH_SETTINGS["tables"], dtype=cp.int32)[None, :, None]

    with np.load(INTERMEDIATE) as encoded:
        vectors = encoded["vectors"]
        for first in range(0, len(vectors), batch_size):
            stop = min(first + batch_size, len(vectors))
            queries = cp.ascontiguousarray(cp.asarray(vectors[first:stop]), dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            q_start = time.perf_counter()
            lsh_tracks, lsh_scores, _ = index.search(queries, probes=LSH_SETTINGS["probes"])
            top1_tracks.append(cp.asnumpy(lsh_tracks))
            top1_scores.append(cp.asnumpy(lsh_scores))
            cp.cuda.Stream.null.synchronize()
            if first > 0:  # the first synchronized batch warms the CUDA context and kernels
                search_seconds += time.perf_counter() - q_start

            short_start = time.perf_counter()
            batch_count = len(queries)
            projections = (queries @ index.planes).reshape(
                batch_count, LSH_SETTINGS["tables"], LSH_SETTINGS["bits"]
            )
            absolute = cp.abs(projections)
            first_bit = absolute.argmin(axis=2)
            second_absolute = absolute.copy()
            cp.put_along_axis(second_absolute, first_bit[..., None], cp.inf, axis=2)
            second_bit = second_absolute.argmin(axis=2)
            signs = projections >= 0
            weights = 1 << cp.arange(LSH_SETTINGS["bits"], dtype=cp.int32)
            keys = (signs * weights).sum(axis=2, dtype=cp.int32)
            masks = cp.stack(
                (
                    cp.zeros_like(first_bit),
                    1 << first_bit,
                    1 << second_bit,
                    (1 << first_bit) | (1 << second_bit),
                ),
                axis=2,
            )
            buckets = keys[:, :, None] ^ masks
            starts = index.offsets[tables, buckets]
            ends = index.offsets[tables, buckets + 1]
            positions = starts[..., None] + cp.arange(maximum_bucket, dtype=cp.int32)
            valid = positions < ends[..., None]
            safe_positions = cp.minimum(positions, len(index.refs) - 1)
            table_ids = cp.broadcast_to(tables[..., None], safe_positions.shape)
            candidate_rows = index.order[table_ids, safe_positions]
            candidate_rows = cp.where(valid, candidate_rows, -1).reshape(batch_count, -1)
            flat_rows = candidate_rows.reshape(-1)
            flat_query_ids = cp.broadcast_to(
                cp.arange(batch_count, dtype=cp.int32)[:, None], candidate_rows.shape
            ).reshape(-1)
            winner_keys = cp.zeros((batch_count * n_groups,), dtype=cp.uint64)
            total = len(flat_rows)
            for offset in range(0, total, chunk_size):
                row_slice = flat_rows[offset : offset + chunk_size]
                valid_slice = row_slice >= 0
                if not bool(cp.any(valid_slice).item()):
                    continue
                rows = row_slice[valid_slice]
                qids = flat_query_ids[offset : offset + len(row_slice)][valid_slice]
                scores = cp.sum(index.refs[rows] * queries[qids], axis=1, dtype=cp.float32)
                destination = qids * n_groups + group_codes[rows]
                raw_bits = scores.view(cp.uint32)
                ordered_bits = cp.where(
                    (raw_bits & cp.uint32(0x80000000)) != 0,
                    ~raw_bits,
                    raw_bits ^ cp.uint32(0x80000000),
                ).astype(cp.uint64)
                packed = (ordered_bits << cp.uint64(32)) | (
                    cp.uint64(0xFFFFFFFF) - rows.astype(cp.uint64)
                )
                cp.maximum.at(winner_keys, destination, packed)

            winner_keys = winner_keys.reshape(batch_count, n_groups)
            ordered_bits = (winner_keys >> cp.uint64(32)).astype(cp.uint32)
            raw_bits = cp.where(
                (ordered_bits & cp.uint32(0x80000000)) != 0,
                ordered_bits ^ cp.uint32(0x80000000),
                ~ordered_bits,
            )
            group_scores = raw_bits.view(cp.float32)
            group_scores = cp.where(winner_keys != 0, group_scores, -cp.inf)
            selected_groups = cp.argpartition(group_scores, -5, axis=1)[:, -5:]
            selected_scores = cp.take_along_axis(group_scores, selected_groups, axis=1)
            order = cp.argsort(-selected_scores, axis=1)
            selected_groups = cp.take_along_axis(selected_groups, order, axis=1)
            selected_scores = cp.take_along_axis(selected_scores, order, axis=1)
            selected_keys = cp.take_along_axis(winner_keys, selected_groups, axis=1)
            selected_rows = (
                cp.uint64(0xFFFFFFFF) - (selected_keys & cp.uint64(0xFFFFFFFF))
            ).astype(cp.int32)
            selected_tracks = tracks[selected_rows]
            if cp.any(~cp.isfinite(selected_scores)).item() or cp.any(selected_rows >= len(tracks)).item():
                print(
                    "shortlist diagnostic",
                    {
                        "batch": first,
                        "reference_rows": len(index.refs),
                        "groups": n_groups,
                        "candidate_rows": int(cp.count_nonzero(candidate_rows >= 0).item()),
                        "unique_candidate_groups": int(cp.count_nonzero(winner_keys > 0).item()),
                        "finite_top_scores": int(cp.count_nonzero(cp.isfinite(selected_scores)).item()),
                        "candidate_group_codes": tuple(map(int, cp.unique(group_codes).get()[:5])),
                        "top1": cp.asnumpy(lsh_tracks)[:3].tolist(),
                    },
                    flush=True,
                )
                raise ValueError("LSH returned fewer than five unique candidate groups")
            top5_groups.append(group_ids[cp.asnumpy(selected_groups)])
            top5_scores.append(cp.asnumpy(selected_scores))
            top5_tracks.append(cp.asnumpy(selected_tracks))
            cp.cuda.Stream.null.synchronize()
            if first > 0:
                shortlist_seconds += time.perf_counter() - short_start
                timed_queries += batch_count
            if stop % 256 == 0 or stop == len(vectors):
                print(f"LSH scored top-five groups for {stop}/{len(vectors)} queries", flush=True)

    result = {
        "candidates": np.concatenate(top5_tracks).astype(np.int32),
        "candidate_group_id": np.concatenate(top5_groups).astype(np.int32),
        "scores": np.concatenate(top5_scores).astype(np.float32),
        "lsh_top1_track_id": np.concatenate(top1_tracks).astype(np.int32),
        "lsh_top1_score": np.concatenate(top1_scores).astype(np.float32),
    }
    assert result["candidates"].shape == (2048, 5)
    assert result["scores"].shape == (2048, 5)
    if np.any(result["candidate_group_id"][:, 1:] == result["candidate_group_id"][:, :1]):
        raise ValueError("candidate list contains duplicate recording groups")
    with np.load(INTERMEDIATE) as encoded:
        query_groups = encoded["group_id"]
        known = encoded["known"]
        for query_index in np.flatnonzero(known):
            if int(query_groups[query_index]) in absent_groups:
                raise ValueError("known recording group overlaps the absent-song exclusion set")
    np.savez(FRESH / "shortlist.npz", **result)
    timing = {
        "retrieval": "pinned random-hyperplane LSH, 10 bits x 64 tables, 3 extra probes",
        "shortlist": "top five distinct duplicate groups, ranked by maximum exact cosine in visited buckets",
        "searchable_reference_windows": len(reference_vectors),
        "searchable_reference_tracks": len(np.unique(reference_tracks)),
        "searchable_reference_groups": len(group_ids),
        "excluded_absent_groups": len(absent_groups),
        "lsh_index_build_seconds": index_build_seconds,
        "timing_warmup_queries": min(batch_size, len(vectors)),
        "timed_queries": timed_queries,
        "lsh_top1_search_seconds": search_seconds,
        "lsh_top1_search_ms_per_query": 1000 * search_seconds / timed_queries,
        "top5_shortlist_seconds": shortlist_seconds,
        "top5_shortlist_ms_per_query": 1000 * shortlist_seconds / timed_queries,
        "query_batch_size": batch_size,
        "candidate_chunk_size": chunk_size,
        "lsh_max_bucket_size": maximum_bucket,
        "lsh_settings": LSH_SETTINGS,
    }
    (FRESH / "shortlist-timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2), flush=True)


def validate_final_output() -> None:
    """Check saved labels, scores and candidate identities without reading 2 GB of audio at once."""
    with np.load(OUTPUT) as result:
        if "audio" not in result.files or result["candidates"].shape != (2048, 5):
            raise ValueError("saved fresh shortlist artifact has an unexpected shape")
        if result["scores"].shape != (2048, 5) or np.any(result["scores"][:, :-1] < result["scores"][:, 1:]):
            raise ValueError("candidate scores are not ordered")
        candidate_groups = result["candidate_group_id"]
        if candidate_groups.shape != (2048, 5) or np.any(np.diff(np.sort(candidate_groups, axis=1), axis=1) == 0):
            raise ValueError("each query needs five distinct candidate groups")
        correct = candidate_groups[:, 0] == result["group_id"]
        top5 = np.any(candidate_groups == result["group_id"][:, None], axis=1)
        for phase_name in ("development", "test"):
            for known_flag in (False, True):
                mask = (result["phase"] == phase_name) & (result["known"] == known_flag)
                print(
                    f"{phase_name} {'known' if known_flag else 'absent'}: "
                    f"top1 {int(np.sum(correct & mask))}/{int(mask.sum())}, "
                    f"top5 {int(np.sum(top5 & mask))}/{int(mask.sum())}"
                )
    with zipfile.ZipFile(OUTPUT) as archive, archive.open("audio.npy") as member:
        version = np.lib.format.read_magic(member)
        if version != (1, 0):
            raise ValueError(f"unsupported audio NPY header version {version}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        if shape != (2048, WINDOW_SAMPLES) or fortran_order or dtype != np.dtype(np.float32):
            raise ValueError("saved audio array has an unexpected shape or dtype")
    print(f"Verified {OUTPUT} without loading its full audio matrix into memory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.search_only:
        search_only()
        return
    if args.package_only:
        package_outputs()
        validate_final_output()
        return
    make_queries()
    run_search_subprocess()
    validate_final_output()


if __name__ == "__main__":
    main()
