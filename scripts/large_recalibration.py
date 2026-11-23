"""Calibrate fixed score-only and score-gap rules on FMA Large development tracks.

Run from the repository root in the project's CuPy environment:
    python scripts/large_recalibration.py

The script uses the existing 2,048-track PR6 query cohort. Every unknown-song
track in both splits is removed from the shared reference catalogue before
scoring. Recalibration sees development scores only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LSH_COMMIT = "5e9427ee5f83cd32fdd9b5a1c8f40396c3772989"
PR6_COMMIT = "c897de3728d60c47fda5271449b8166aa09cd01b"
PR7_COMMIT = "d0be3706a8b4a0b7c62e5c60b631fd3fd1abf618"
LSH_SETTINGS = {"bits": 10, "tables": 64, "probes": 3, "seed": 2807}
FAR_TARGET = 0.01
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "data/medium/cupy-cache"))


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"])


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def load_pinned_lsh(temp_root: Path):
    package = temp_root / "pinned_lsh"
    package.mkdir()
    for filename in ("gpu_retrieval.py", "gpu_lsh.cu"):
        (package / filename).write_bytes(git_blob(LSH_COMMIT, f"src/afp/{filename}"))
    spec = importlib.util.spec_from_file_location("large_calibration_lsh", package / "gpu_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_archives(reference_path: Path, query_path: Path):
    import cupy as cp

    with np.load(reference_path) as archive:
        reference_vectors = np.asarray(archive["vectors"], dtype=np.float32)
        reference_tracks = np.asarray(archive["track_id"], dtype=np.int32)
    catalogue = json.loads(reference_path.with_name("catalogue.json").read_text(encoding="utf-8"))
    reference_sha256 = digest(reference_path)
    if reference_sha256 != catalogue["references_sha256"]:
        raise ValueError("The Large catalogue does not match its recorded fingerprint checksum")
    with np.load(query_path) as archive:
        query = {name: archive[name].copy() for name in archive.files}

    if set(query["phase"]) != {"development", "test"}:
        raise ValueError("Expected the original PR6 development and test splits")
    if set(query["family"]) != {"standard", "tiktok"}:
        raise ValueError("Expected standard and TikTok-like query families")
    if set(query["known"].tolist()) != {False, True}:
        raise ValueError("Expected both known-song and missing-song clips")
    if len(np.unique(query["track_id"])) != 2048:
        raise ValueError("The PR6 cohort must retain its 2,048 distinct songs")

    previous = json.loads(git_blob(PR6_COMMIT, "reports/runner-up-margin.json"))
    if digest(query_path) != previous["data"]["query_embeddings_sha256"]:
        raise ValueError("The local query archive does not match the released PR6 query set")
    provenance = list(
        csv.DictReader(io.StringIO(git_blob(PR6_COMMIT, "reports/runner-up-margin-cohort-tracks.csv").decode("utf-8")))
    )
    cohort_ids = [int(row["track_id"]) for row in provenance if row["cohort"] == "runner_up"]
    if len(cohort_ids) != 2048 or set(cohort_ids) != set(map(int, np.unique(query["track_id"]))):
        raise ValueError("The query tracks differ from the frozen PR6 cohort")
    for track in np.unique(query["track_id"]):
        rows = query["track_id"] == track
        if len(np.unique(query["phase"][rows])) != 1 or len(np.unique(query["known"][rows])) != 1:
            raise ValueError(f"Queries for song {track} span inconsistent split or known status")

    removed_tracks = np.unique(query["track_id"][~query["known"]])
    keep = ~np.isin(reference_tracks, removed_tracks)
    reference_vectors = reference_vectors[keep]
    reference_tracks = reference_tracks[keep]
    if not np.array_equal(np.isin(query["track_id"], reference_tracks), query["known"]):
        raise ValueError("Known labels do not match catalogue membership after removing unknown songs")

    return cp, query, reference_vectors, reference_tracks, previous, catalogue


def similarity_to_track_scores(cp, similarities, track_ids, n_tracks):
    by_track = cp.full((len(similarities), n_tracks), -cp.inf, dtype=cp.float32)
    destination = cp.arange(len(similarities), dtype=cp.int64)[:, None] * n_tracks + track_ids[None, :]
    cp.maximum.at(by_track.ravel(), destination.ravel(), similarities.ravel())
    return by_track


def lsh_candidate_mask(cp, index, queries):
    projections = (queries @ index.planes).reshape(len(queries), LSH_SETTINGS["tables"], LSH_SETTINGS["bits"])
    absolute = cp.abs(projections)
    first = absolute.argmin(axis=2)
    second_absolute = absolute.copy()
    cp.put_along_axis(second_absolute, first[:, :, None], cp.inf, axis=2)
    second = second_absolute.argmin(axis=2)
    signs = projections >= 0
    weights = 1 << cp.arange(LSH_SETTINGS["bits"], dtype=cp.int32)
    keys = (signs * weights).sum(axis=2, dtype=cp.int32)
    masks = cp.stack(
        (cp.zeros_like(first), 1 << first, 1 << second, (1 << first) | (1 << second)),
        axis=2,
    )
    buckets = keys[:, :, None] ^ masks
    table_ids = cp.arange(LSH_SETTINGS["tables"], dtype=cp.int32)[None, :, None]
    starts = index.offsets[table_ids, buckets]
    ends = index.offsets[table_ids, buckets + 1]
    maximum_bucket_size = int(cp.max(ends - starts).item())
    positions = starts[..., None] + cp.arange(maximum_bucket_size, dtype=cp.int32)
    valid = positions < ends[..., None]
    safe_positions = cp.minimum(positions, len(index.refs) - 1)
    candidate_rows = index.order[table_ids[..., None], safe_positions]
    query_ids = cp.broadcast_to(cp.arange(len(queries), dtype=cp.int32)[:, None, None, None], candidate_rows.shape)
    mask = cp.zeros((len(queries), len(index.refs)), dtype=cp.bool_)
    mask[query_ids[valid], candidate_rows[valid]] = True
    return mask


def score_queries(cp, lsh_module, references_np, reference_tracks_np, query, batch_size):
    cp.cuda.Device(0).use()
    references = cp.asarray(references_np, dtype=cp.float32)
    track_ids = cp.asarray(reference_tracks_np, dtype=cp.int32)
    query_vectors = query["vectors"]
    if not np.isfinite(query_vectors).all():
        raise ValueError("Query fingerprints contain nonfinite values")

    index = lsh_module.GPUAngularLSH(
        references,
        track_ids,
        bits=LSH_SETTINGS["bits"],
        tables=LSH_SETTINGS["tables"],
        seed=LSH_SETTINGS["seed"],
    )
    n_tracks = int(reference_tracks_np.max()) + 1
    outputs = {
        method: {name: [] for name in ("top_track", "top_score", "runner_track", "runner_score")}
        for method in ("exact", "lsh")
    }
    lsh_top_one_matches = 0
    for first in range(0, len(query_vectors), batch_size):
        stop = min(first + batch_size, len(query_vectors))
        q = cp.asarray(query_vectors[first:stop], dtype=cp.float32)
        similarities = q @ references.T
        exact_by_track = similarity_to_track_scores(cp, similarities, track_ids, n_tracks)
        exact_winner, exact_score = lsh_module.gpu_exact(references, track_ids, q)
        rows = cp.arange(len(q), dtype=cp.int64)
        exact_runners = exact_by_track.copy()
        exact_runners[rows, exact_winner] = -cp.inf
        exact_runner = exact_runners.argmax(axis=1)
        exact_runner_score = exact_runners[rows, exact_runner]
        outputs["exact"]["top_track"].append(cp.asnumpy(exact_winner))
        outputs["exact"]["top_score"].append(cp.asnumpy(exact_score))
        outputs["exact"]["runner_track"].append(cp.asnumpy(exact_runner))
        outputs["exact"]["runner_score"].append(cp.asnumpy(exact_runner_score))

        lsh_winner, lsh_score, _ = index.search(q, probes=LSH_SETTINGS["probes"])
        candidates = lsh_candidate_mask(cp, index, q)
        candidate_by_track = similarity_to_track_scores(
            cp, cp.where(candidates, similarities, -cp.inf), track_ids, n_tracks
        )
        candidate_winner_score = candidate_by_track[rows, lsh_winner]
        delta = cp.abs(candidate_winner_score - lsh_score)
        if cp.any(delta > 2e-5).item():
            raise ValueError("Reconstructed LSH buckets do not contain the pinned winner")
        lsh_top_one_matches += int(cp.count_nonzero(delta <= 2e-5).item())
        lsh_runners = candidate_by_track.copy()
        lsh_runners[rows, lsh_winner] = -cp.inf
        lsh_runner = lsh_runners.argmax(axis=1)
        lsh_runner_score = lsh_runners[rows, lsh_runner]
        outputs["lsh"]["top_track"].append(cp.asnumpy(lsh_winner))
        outputs["lsh"]["top_score"].append(cp.asnumpy(lsh_score))
        outputs["lsh"]["runner_track"].append(cp.asnumpy(lsh_runner))
        outputs["lsh"]["runner_score"].append(cp.asnumpy(lsh_runner_score))
        del similarities, exact_by_track, exact_runners, candidate_by_track, lsh_runners

    result = {}
    for method, fields in outputs.items():
        result[method] = {name: np.concatenate(values) for name, values in fields.items()}
        result[method]["gap"] = np.maximum(result[method]["top_score"] - result[method]["runner_score"], 0)
        if np.any(result[method]["top_track"] == result[method]["runner_track"]):
            raise ValueError(f"{method} returned the same song as both top result and runner-up")
    result["_lsh_top_one_matches"] = lsh_top_one_matches
    return result


def choose_score_rule(dev, budget):
    unknown = ~dev["known"]
    correct = dev["known"] & (dev["top_track"] == dev["track_id"])
    scores = dev["top_score"].astype(np.float64)
    maximum_unknown = np.float64(scores[unknown].max())
    thresholds = np.unique(np.r_[scores, np.nextafter(maximum_unknown, np.float64(np.inf))])
    best = None
    for threshold in thresholds:
        false_accepts = int(np.count_nonzero(unknown & (scores >= threshold)))
        if false_accepts > budget:
            continue
        correct_accepts = int(np.count_nonzero(correct & (scores >= threshold)))
        key = (correct_accepts, -false_accepts, float(threshold))
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "score_threshold": float(threshold),
                    "gap_threshold": None,
                    "development_known_correct": correct_accepts,
                    "development_unknown_false_accepts": false_accepts,
                },
            )
    if best is None:
        raise RuntimeError("No score-only threshold satisfies the development FAR limit")
    return best[1]


def choose_gap_rule(dev, budget):
    unknown = ~dev["known"]
    correct = dev["known"] & (dev["top_track"] == dev["track_id"])
    scores = dev["top_score"].astype(np.float64)
    gaps = dev["gap"].astype(np.float64)
    maximum_unknown = np.float64(scores[unknown].max())
    thresholds = np.unique(np.r_[scores, np.nextafter(maximum_unknown, np.float64(np.inf))])
    best = None
    for score_threshold in thresholds:
        candidate = gaps[unknown & (scores >= score_threshold)]
        gap_threshold = (
            np.nextafter(np.float64(np.sort(candidate)[::-1][budget]), np.float64(np.inf))
            if len(candidate) > budget
            else -np.inf
        )
        false_accepts = int(np.count_nonzero(unknown & (scores >= score_threshold) & (gaps >= gap_threshold)))
        correct_accepts = int(np.count_nonzero(correct & (scores >= score_threshold) & (gaps >= gap_threshold)))
        key = (correct_accepts, -false_accepts, float(score_threshold), float(gap_threshold))
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "score_threshold": float(score_threshold),
                    "gap_threshold": None if np.isneginf(gap_threshold) else float(gap_threshold),
                    "development_known_correct": correct_accepts,
                    "development_unknown_false_accepts": false_accepts,
                },
            )
    if best is None:
        raise RuntimeError("No score-and-gap thresholds satisfy the development FAR limit")
    return best[1]


def decisions(rows, rule):
    accept = rows["top_score"].astype(np.float64) >= rule["score_threshold"]
    if rule["gap_threshold"] is not None:
        accept &= rows["gap"].astype(np.float64) >= rule["gap_threshold"]
    correct = accept & rows["known"] & (rows["top_track"] == rows["track_id"])
    return accept, correct


def paired_track_interval(rows, event_a, event_b, seed, reps=10_000):
    difference = event_a.astype(np.float64) - event_b.astype(np.float64)
    tracks = np.unique(rows["track_id"])
    per_track = np.array([difference[rows["track_id"] == tid].mean() for tid in tracks])
    sampled = np.random.default_rng(seed).integers(0, len(per_track), size=(reps, len(per_track)))
    rates = per_track[sampled].mean(axis=1)
    return [float(x) for x in np.quantile(rates, [0.025, 0.975])]


def track_bootstrap_rate(rows, events, seed):
    per_track_rate = np.array(
        [events[rows["track_id"] == tid].mean() for tid in np.unique(rows["track_id"])],
        dtype=np.float64,
    )
    rate = float(per_track_rate.mean())
    if not np.isclose(rate, events.mean(), rtol=0, atol=1e-12):
        raise ValueError("The song bootstrap center does not match its query-level rate")
    interval = paired_track_interval(rows, events, np.zeros(len(events), dtype=bool), seed)
    return {
        "rate": rate,
        "interval_95": interval,
        "midpoint_error": abs((interval[0] + interval[1]) / 2 - rate),
    }


def evaluate(rows, rules, phase, seed_base):
    selection = rows["phase"] == phase
    subset = {name: value[selection] for name, value in rows.items()}
    output = {}
    for r_index, (rule_name, rule) in enumerate(rules.items()):
        accept, correct = decisions(subset, rule)
        known, unknown = subset["known"], ~subset["known"]
        family_results = {}
        for family in ("standard", "tiktok"):
            family_mask = subset["family"] == family
            family_known = known & family_mask
            family_unknown = unknown & family_mask
            family_results[family] = {
                "known_correct_accepted": int(np.count_nonzero(correct & family_known)),
                "known_queries": int(np.count_nonzero(family_known)),
                "unknown_false_accepts": int(np.count_nonzero(accept & family_unknown)),
                "unknown_queries": int(np.count_nonzero(family_unknown)),
            }
        output[rule_name] = {
            "score_threshold": rule["score_threshold"],
            "gap_threshold": rule["gap_threshold"],
            "development_known_correct": rule.get("development_known_correct"),
            "development_unknown_false_accepts": rule.get("development_unknown_false_accepts"),
            "known_correct_accepted": int(np.count_nonzero(correct & known)),
            "known_accepted_incorrectly": int(np.count_nonzero(accept & known & ~correct)),
            "known_queries": int(known.sum()),
            "unknown_false_accepts": int(np.count_nonzero(accept & unknown)),
            "unknown_queries": int(unknown.sum()),
            "by_family": family_results,
        }
        if phase == "test":
            known_rows = {name: value[known] for name, value in subset.items()}
            unknown_rows = {name: value[~known] for name, value in subset.items()}
            known_rate = track_bootstrap_rate(known_rows, correct[known], seed_base + r_index)
            unknown_rate = track_bootstrap_rate(unknown_rows, accept[unknown], seed_base + 20 + r_index)
            output[rule_name]["known_correct_acceptance_rate"] = known_rate["rate"]
            output[rule_name]["unknown_false_acceptance_rate"] = unknown_rate["rate"]
            output[rule_name]["track_bootstrap_95"] = {
                "known_correct_acceptance_rate": known_rate["interval_95"],
                "unknown_false_acceptance_rate": unknown_rate["interval_95"],
            }
            output[rule_name]["bootstrap_midpoint_error"] = {
                "known_correct_acceptance_rate": known_rate["midpoint_error"],
                "unknown_false_acceptance_rate": unknown_rate["midpoint_error"],
            }
    return subset, output


def compare_with_frozen(rows, calibrated, frozen):
    _, calibrated_result = evaluate(rows, calibrated, "test", 55800)
    _, frozen_result = evaluate(rows, frozen, "test", 55900)
    subset = {name: value[rows["phase"] == "test"] for name, value in rows.items()}
    comparisons = {}
    for rule_name in calibrated:
        cal_accept, cal_correct = decisions(subset, calibrated[rule_name])
        old_accept, old_correct = decisions(subset, frozen[rule_name])
        known, unknown = subset["known"], ~subset["known"]
        paired_change = cal_correct.astype(float) - old_correct.astype(float)
        comparisons[rule_name] = {
            "frozen": frozen_result[rule_name],
            "large_recalibrated": calibrated_result[rule_name],
            "known_correct_retained": int(np.count_nonzero(cal_correct & old_correct)),
            "known_correct_lost": int(np.count_nonzero(old_correct & ~cal_correct)),
            "known_correct_gained": int(np.count_nonzero(cal_correct & ~old_correct)),
            "known_correct_net_change": int(np.count_nonzero(cal_correct) - np.count_nonzero(old_correct)),
            "paired_known_correct_change_percentage_points": float(100 * paired_change[known].mean()),
            "paired_known_correct_track_bootstrap_95_percentage_points": [
                100 * x
                for x in paired_track_interval(
                    {name: value[subset["known"]] for name, value in subset.items()},
                    cal_correct[known],
                    old_correct[known],
                    56000 + len(comparisons),
                )
            ],
            "unknown_false_accept_change": int(
                np.count_nonzero(cal_accept & unknown) - np.count_nonzero(old_accept & unknown)
            ),
            "by_family": {},
        }
        for family in ("standard", "tiktok"):
            mask = subset["family"] == family
            family_known = mask & known
            family_unknown = mask & unknown
            comparisons[rule_name]["by_family"][family] = {
                "frozen_known_correct": int(np.count_nonzero(old_correct & family_known)),
                "recalibrated_known_correct": int(np.count_nonzero(cal_correct & family_known)),
                "frozen_unknown_false_accepts": int(np.count_nonzero(old_accept & family_unknown)),
                "recalibrated_unknown_false_accepts": int(np.count_nonzero(cal_accept & family_unknown)),
            }
    return comparisons


def write_scores(path, query, scored):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = (
        "method",
        "track_id",
        "known",
        "phase",
        "family",
        "distortion",
        "top_track",
        "top_score",
        "runner_track",
        "runner_score",
        "gap",
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as compressed:
        writer = csv.DictWriter(compressed, fieldnames=names)
        writer.writeheader()
        for method in ("exact", "lsh"):
            for index in range(len(query["track_id"])):
                writer.writerow(
                    {
                        "method": method,
                        "track_id": int(query["track_id"][index]),
                        "known": bool(query["known"][index]),
                        "phase": str(query["phase"][index]),
                        "family": str(query["family"][index]),
                        "distortion": str(query["distortion"][index]),
                        **{
                            field: (
                                int(scored[method][field][index])
                                if field.endswith("track")
                                else float(scored[method][field][index])
                            )
                            for field in ("top_track", "top_score", "runner_track", "runner_score", "gap")
                        },
                    }
                )


def analyse_method(query, scores, frozen_rules, budget):
    rows = {
        "track_id": query["track_id"],
        "known": query["known"],
        "phase": query["phase"],
        "family": query["family"],
        **scores,
    }
    development = {name: value[query["phase"] == "development"] for name, value in rows.items()}
    recalibrated = {
        "score_only": choose_score_rule(development, budget),
        "score_and_gap": choose_gap_rule(development, budget),
    }
    frozen = frozen_rules
    # Keep the development summaries' selected parameters alongside the FAR results.
    dev_results = {}
    dev_data = {name: value[query["phase"] == "development"] for name, value in rows.items()}
    for rule_name, rule in recalibrated.items():
        accepted, correct = decisions(dev_data, rule)
        dev_results[rule_name] = {
            **rule,
            "known_correct_accepted": int(np.count_nonzero(correct & dev_data["known"])),
            "unknown_false_accepts": int(np.count_nonzero(accepted & ~dev_data["known"])),
            "known_queries": int(dev_data["known"].sum()),
            "unknown_queries": int((~dev_data["known"]).sum()),
        }
    _, frozen_dev = evaluate(rows, frozen, "development", 55700)
    _, recal_test = evaluate(rows, recalibrated, "test", 55800)
    _, frozen_test = evaluate(rows, frozen, "test", 55900)
    return rows, {
        "calibration": {"frozen": frozen_dev, "large_development_recalibrated": dev_results},
        "test": {"frozen": frozen_test, "large_development_recalibrated": recal_test},
        "comparison": compare_with_frozen(rows, recalibrated, frozen),
    }


def render_report(result):
    lines = [
        "# Recalibrating unknown-song thresholds on FMA Large",
        "",
        f"Reference: {result['data']['reference_tracks']:,} tracks, {result['data']['reference_windows']:,} windows after removing the 1,024 unknown cohort tracks.",
        f"Query cohort: {result['data']['queries']:,} clips from the original 2,048-song PR6 cohort. Development/test splits are unchanged.",
        f"Calibration allowed {result['calibration']['false_accept_budget']} false accepts among {result['calibration']['unknown_development_queries']:,} Large development unknown clips.",
        "",
        "## Held-out known-song results",
        "",
        "| Retrieval | Rule | Frozen correct | Large recalibrated correct | Retained | Lost | Gained | Net change | Δ correct pp (95% song CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("exact", "lsh"):
        comparison = result["methods"][method]["comparison"]
        for rule in ("score_only", "score_and_gap"):
            row = comparison[rule]
            old = row["frozen"]["known_correct_accepted"]
            new = row["large_recalibrated"]["known_correct_accepted"]
            change = row["paired_known_correct_change_percentage_points"]
            low, high = row["paired_known_correct_track_bootstrap_95_percentage_points"]
            lines.append(
                f"| {method.upper()} | {rule.replace('_', ' ')} | {old}/1,024 | {new}/1,024 | "
                f"{row['known_correct_retained']} | {row['known_correct_lost']} | "
                f"{row['known_correct_gained']} | {row['known_correct_net_change']:+d} | "
                f"{change:+.1f} ({low:+.1f}, {high:+.1f}) |"
            )
    lines.extend(
        [
            "",
            "## Large development calibration",
            "",
            "| Retrieval | Rule | Frozen known correct | Frozen false accepts | Recalibrated known correct | Recalibrated false accepts |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("exact", "lsh"):
        calibration = result["methods"][method]["calibration"]
        for rule in ("score_only", "score_and_gap"):
            frozen = calibration["frozen"][rule]
            recalibrated = calibration["large_development_recalibrated"][rule]
            lines.append(
                f"| {method.upper()} | {rule.replace('_', ' ')} | "
                f"{frozen['known_correct_accepted']}/1,024 | "
                f"{frozen['unknown_false_accepts']}/1,024 | "
                f"{recalibrated['known_correct_accepted']}/1,024 | "
                f"{recalibrated['unknown_false_accepts']}/1,024 |"
            )
    lines.extend(
        [
            "",
            "## Thresholds learned from development",
            "",
            "| Retrieval | Rule | Score threshold | Gap threshold | Development known correct | Development unknown false accepts |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("exact", "lsh"):
        for rule, row in result["methods"][method]["calibration"]["large_development_recalibrated"].items():
            gap = "—" if row["gap_threshold"] is None else f"{row['gap_threshold']:.9f}"
            lines.append(
                f"| {method.upper()} | {rule.replace('_', ' ')} | {row['score_threshold']:.9f} | "
                f"{gap} | {row['known_correct_accepted']}/1,024 | {row['unknown_false_accepts']}/1,024 |"
            )
    lines.extend(
        [
            "",
            "## Held-out unknown-song results",
            "",
            "| Retrieval | Rule | Frozen false accepts | Recalibrated false accepts |",
            "|---|---|---:|---:|",
        ]
    )
    for method in ("exact", "lsh"):
        for rule, row in result["methods"][method]["comparison"].items():
            lines.append(
                f"| {method.upper()} | {rule.replace('_', ' ')} | "
                f"{row['frozen']['unknown_false_accepts']}/1,024 | "
                f"{row['large_recalibrated']['unknown_false_accepts']}/1,024 |"
            )
    lines.extend(
        [
            "",
            "Calibration maximises correctly accepted known clips on development tracks subject to an empirical 1% unknown false-accept budget. The held-out test contains 1,024 known and 1,024 missing-song clips. Thresholds were selected without test labels.",
            "",
            "For both retrievers, the best Large-development score-and-gap operating point selected no gap threshold. The recalibrated joint rule therefore coincides with score-only. Both recalibrated rules produced 8 false accepts among 1,024 held-out missing-song clips.",
            "",
            "## Distortion-family breakdown",
            "",
            "| Retrieval | Rule | Family | Frozen known correct | Recalibrated known correct | Frozen unknown false accepts | Recalibrated unknown false accepts |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("exact", "lsh"):
        for rule, row in result["methods"][method]["comparison"].items():
            for family, family_row in row["by_family"].items():
                lines.append(
                    f"| {method.upper()} | {rule.replace('_', ' ')} | {family} | "
                    f"{family_row['frozen_known_correct']} | {family_row['recalibrated_known_correct']} | "
                    f"{family_row['frozen_unknown_false_accepts']} | "
                    f"{family_row['recalibrated_unknown_false_accepts']} |"
                )
    lines.extend(
        [
            "",
            "Paired confidence intervals resample songs and keep each song's standard and TikTok-like queries together. Bootstrap point estimates were checked against the query-level rates.",
            "",
            "## Reproduction",
            "",
            "Run `python scripts/large_recalibration.py` from the repository root in its CuPy environment. The Large reference comes from the FMA Large build described in PR7; the query archive and ordered cohort are the saved PR6 inputs. A fresh clone also needs the pinned PR source objects:",
            "",
            "~~~powershell",
            "git fetch https://github.com/nickiliak/02807_computational_tools_ds.git refs/pull/1/head refs/pull/6/head refs/pull/7/head",
            "python scripts/large_recalibration.py",
            "~~~",
            "",
            "The script checks the Large fingerprint checksum, PR6 query checksum and cohort IDs, and dynamically loads the LSH implementation from its pinned source commit. It writes query-level scores to the compressed CSV, along with the thresholds and song-bootstrap summaries in JSON.",
        ]
    )
    return "\n".join(lines)


def run(args):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="large-recalibration-") as work:
        lsh_module = load_pinned_lsh(Path(work))
        cp, query, references, reference_tracks, pr6, catalogue = load_archives(args.reference, args.queries)
        print(
            f"Scoring {len(query['track_id']):,} original PR6 queries against "
            f"{len(reference_tracks):,} Large windows on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
            flush=True,
        )
        scores = score_queries(cp, lsh_module, references, reference_tracks, query, args.batch_size)

    pr7 = json.loads(git_blob(PR7_COMMIT, "reports/catalogue-growth.json"))
    frozen_rules = {
        "score_only": {
            "score_threshold": pr6["threshold_selection"]["maximum_score"]["score_threshold"],
            "gap_threshold": None,
        },
        "score_and_gap": {
            "score_threshold": pr6["threshold_selection"]["score_and_margin"]["score_threshold"],
            "gap_threshold": pr6["threshold_selection"]["score_and_margin"]["gap_threshold"],
        },
    }
    frozen_thresholds = pr7["protocol"]["open_set"]["thresholds"]
    if frozen_thresholds["maximum_score"]["score_threshold"] != frozen_rules["score_only"]["score_threshold"]:
        raise ValueError("Frozen score threshold differs from the reviewed Medium threshold")
    if frozen_thresholds["score_and_margin"]["score_threshold"] != frozen_rules["score_and_gap"]["score_threshold"]:
        raise ValueError("Frozen joint score threshold differs from the reviewed Medium threshold")
    if frozen_thresholds["score_and_margin"]["gap_threshold"] != frozen_rules["score_and_gap"]["gap_threshold"]:
        raise ValueError("Frozen gap threshold differs from the reviewed Medium threshold")

    budget = int(np.floor(FAR_TARGET * np.count_nonzero((query["phase"] == "development") & ~query["known"])))
    if budget != 10:
        raise ValueError(f"Expected the PR6 10/1,024 development FAR budget, got {budget}")
    methods = {}
    expected_frozen_test = {
        "exact": {"score_only": (523, 18), "score_and_gap": (691, 23)},
        "lsh": {"score_only": (526, 18), "score_and_gap": (691, 23)},
    }
    for method in ("exact", "lsh"):
        _, results = analyse_method(query, scores[method], frozen_rules, budget)
        for rule_name, (known_correct, unknown_false) in expected_frozen_test[method].items():
            measured = results["test"]["frozen"][rule_name]
            if (
                measured["known_correct_accepted"] != known_correct
                or measured["unknown_false_accepts"] != unknown_false
            ):
                raise ValueError(
                    f"{method} frozen {rule_name} does not match the PR7 Large result: "
                    f"{measured['known_correct_accepted']}/{measured['unknown_false_accepts']} "
                    f"versus {known_correct}/{unknown_false}"
                )
        methods[method] = results
    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "data": {
            "reference_sha256": digest(args.reference),
            "reference_tracks": int(np.unique(reference_tracks).size),
            "reference_windows": len(reference_tracks),
            "query_archive_sha256": digest(args.queries),
            "query_embeddings_sha256": pr6["data"]["query_embeddings_sha256"],
            "projection_head_sha256": pr6["data"]["projection_head_sha256"],
            "large_archive_sha1": catalogue["archive_sha1"],
            "queries": len(query["track_id"]),
            "songs": int(np.unique(query["track_id"]).size),
            "development_known_queries": int(np.count_nonzero((query["phase"] == "development") & query["known"])),
            "development_unknown_queries": int(np.count_nonzero((query["phase"] == "development") & ~query["known"])),
            "test_known_queries": int(np.count_nonzero((query["phase"] == "test") & query["known"])),
            "test_unknown_queries": int(np.count_nonzero((query["phase"] == "test") & ~query["known"])),
            "unknown_tracks_removed": int(np.unique(query["track_id"][~query["known"]]).size),
            "query_cohort_sha256": pr6["data"]["cohort_track_ids_sha256"],
            "no_development_test_track_overlap": not bool(
                set(query["track_id"][query["phase"] == "development"])
                & set(query["track_id"][query["phase"] == "test"])
            ),
        },
        "calibration": {
            "target_unknown_false_accept_rate": FAR_TARGET,
            "false_accept_budget": budget,
            "unknown_development_queries": int(np.count_nonzero((query["phase"] == "development") & ~query["known"])),
            "lsh_settings": LSH_SETTINGS,
            "lsh_top_one_score_match_with_bucket_union": scores["_lsh_top_one_matches"],
        },
        "frozen_thresholds": frozen_rules,
        "methods": methods,
        "timing_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Threshold calibration reuses the original PR6 synthetic query cohort and split.",
            "The held-out test sample has 1,024 unknown clips, so 1% corresponds to about ten false accepts and remains a noisy empirical estimate.",
            "LSH runner-up scores come only from the union of its base and three multi-probe buckets in each table.",
            "No real phone recordings or out-of-distribution distortion families are included.",
        ],
    }
    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md = args.output_md
    output_md.write_text(render_report(result), encoding="utf-8")
    write_scores(args.output_scores, query, scores)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/large/references.npz"))
    parser.add_argument("--queries", type=Path, default=Path("data/medium/runner-up-queries.npz"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/large-recalibration.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/large-recalibration.md"))
    parser.add_argument("--output-scores", type=Path, default=Path("reports/large-recalibration-scores.csv.gz"))
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
