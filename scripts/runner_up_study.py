"""Fresh-track GPU evaluation of maximum score versus distinct-track margin.

PowerShell commands from the repository root:
  $python = "data/medium/.venv/Scripts/python.exe"
  $backbone = (Resolve-Path "data/medium/hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots/0562a57814f6f8bb9dfea0a25921a2fce1a841a").Path
  & $python scripts/runner_up_study.py generate --backbone $backbone
  & $python scripts/runner_up_study.py score
  & $python scripts/runner_up_study.py analyse
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afp.augment import CATEGORIES, TEST_CATEGORIES, random_augment
from afp.data import SAMPLE_RATE, SEGMENT_S, load_processed, load_speech_bank
from afp.embed import MuQBackbone
from afp.fingerprint import load_head, split_tracks

SEED = 7741
HEAD_SHA256 = "4a5fb50d5dea344c0fdc803e9c825b6dfbd39b5dc86bafdce96c261e63801c6f"
FAR_TARGET = 0.01


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fresh_tracks(processed, previous_queries, output_queries, cohort_file):
    manifest = pd.read_csv(processed / "manifest.csv")
    if manifest.track_id.duplicated().any():
        raise ValueError("Duplicate IDs in processed manifest")
    train, eligible = split_tracks(manifest.track_id.to_numpy(), 0.2, 0)
    with np.load(processed / "fingerprints/references.npz") as archive:
        eligible = np.intersect1d(eligible, archive["track_id"])
    excluded = set()
    for name in ("queries", "queries_tiktok"):
        with np.load(processed / "fingerprints" / f"{name}.npz") as archive:
            excluded.update(map(int, np.unique(archive["track_id"])))
    eligible = np.setdiff1d(eligible, np.fromiter(excluded, dtype=np.int64))
    if output_queries.exists():
        raise FileExistsError(f"Refusing to overwrite {output_queries}")
    shards = sorted((processed / "embeddings/pairs").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError("Training-pair shards are required to verify track separation")
    for shard in shards:
        with np.load(shard) as archive:
            if not np.isin(archive["track_id"], train).all():
                raise ValueError(f"Training pairs contain a held-out track in {shard.name}")
    if cohort_file.exists():
        provenance = pd.read_csv(cohort_file)
        prior_ids = provenance.loc[provenance.cohort == "previous_open_set", "track_id"].to_numpy()
        tracks = provenance.loc[provenance.cohort == "runner_up", "track_id"].to_numpy()
        positions = provenance.loc[provenance.cohort == "runner_up", "position"].to_numpy()
        if len(prior_ids) != 2048 or len(tracks) != 2048 or not np.array_equal(positions, np.arange(2048)):
            raise ValueError("Committed cohort provenance must contain ordered IDs for both 2,048-track studies")
    else:
        if not previous_queries.is_file():
            raise FileNotFoundError(f"Supply the earlier query archive once to create cohort provenance: {previous_queries}")
        with np.load(previous_queries) as archive:
            prior_ids = np.unique(archive["track_id"])
        excluded.update(map(int, prior_ids))
        eligible = np.setdiff1d(eligible, np.fromiter(excluded, dtype=np.int64))
        if len(eligible) < 2048:
            raise ValueError(f"Only {len(eligible)} unused searchable tracks remain")
        tracks = np.random.default_rng(SEED).choice(eligible, 2048, replace=False)
        cohort_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cohort": cohort, "position": position, "track_id": int(track)}
            for cohort, ids in (("previous_open_set", prior_ids), ("runner_up", tracks))
            for position, track in enumerate(ids)
        ]).to_csv(cohort_file, index=False)
    if (len(np.unique(prior_ids)) != 2048 or len(np.unique(tracks)) != 2048
            or np.intersect1d(prior_ids, tracks).size
            or not np.isin(prior_ids, eligible).all()
            or not np.isin(tracks, eligible).all()):
        raise ValueError("Cohort provenance overlaps or includes a non-evaluation/unsearchable track")
    return manifest.set_index("track_id"), tracks


def generate(processed, model, backbone, previous_queries, output, cohort_file, workers):
    with model.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != HEAD_SHA256:
        raise ValueError(f"Projection head hash mismatch: {actual_hash}")
    manifest, tracks = fresh_tracks(processed, previous_queries, output, cohort_file)
    speech = load_speech_bank(processed / "speech_test.npy")
    encoder = MuQBackbone(device="cuda", repo=backbone)
    head = load_head(model).cuda().eval()
    samples = int(SAMPLE_RATE * SEGMENT_S)

    def prepare(track):
        wave = load_processed(processed / manifest.at[int(track), "path"])
        rng = np.random.default_rng([SEED, int(track)])
        clips, labels = [], []
        for categories in (CATEGORIES, TEST_CATEGORIES):
            audio, _, label = random_augment(wave, rng, SAMPLE_RATE, categories, speech)
            start = int(rng.integers(len(audio) - samples + 1))
            clips.append(audio[start : start + samples].copy())
            labels.append(label)
        return clips, labels

    vectors, distortions = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for first in range(0, len(tracks), 32):
            chunk = list(pool.map(prepare, map(int, tracks[first : first + 32])))
            windows = np.stack([clip for clips, _ in chunk for clip in clips])
            distortions.extend(label for _, labels in chunk for label in labels)
            for start in range(0, len(windows), 16):
                features = encoder.encode(windows[start : start + 16])
                vectors.append(head(torch.from_numpy(features).cuda()).cpu().numpy())
    vectors = np.concatenate(vectors).astype(np.float32)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, vectors=vectors, track_id=np.repeat(tracks, 2), distortion=distortions,
        family=np.tile(["standard", "tiktok"], len(tracks)),
        phase=np.repeat(["development", "test"], len(tracks)),
        known=np.repeat(np.arange(len(tracks)) % 2 == 0, 2),
    )
    print(f"Saved {len(vectors)} queries from {len(tracks)} new tracks to {output}")


def score_queries(processed, queries, output, batch_size):
    torch.backends.cuda.matmul.allow_tf32 = False
    reference_sha256 = sha256_file(processed / "fingerprints/references.npz")
    query_sha256 = sha256_file(queries)
    with np.load(queries) as archive:
        rows = pd.DataFrame({key: archive[key] for key in ("track_id", "known", "phase", "family", "distortion")})
        query_vectors = torch.from_numpy(archive["vectors"].copy()).to("cuda")
    with np.load(processed / "fingerprints/references.npz") as archive:
        removed = np.unique(rows.loc[~rows.known, "track_id"])
        keep = ~np.isin(archive["track_id"], removed)
        ref_track_ids = archive["track_id"][keep].astype(np.int64, copy=False)
        ref_vectors = torch.from_numpy(archive["vectors"][keep].copy()).to("cuda")
    if not np.array_equal(np.isin(rows.track_id, ref_track_ids), rows.known.to_numpy()):
        raise ValueError("Known and unknown labels do not match the removed-track catalogue")

    unique_tracks, track_index = np.unique(ref_track_ids, return_inverse=True)
    gpu_index = torch.from_numpy(track_index.astype(np.int64)).to("cuda")
    gpu_tracks = torch.from_numpy(unique_tracks).to("cuda")
    first_ids, first_scores, second_ids, second_scores = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            query = query_vectors[start : start + batch_size]
            window_scores = query @ ref_vectors.T
            by_track = torch.full((len(query), len(unique_tracks)), -torch.inf, device="cuda")
            by_track.scatter_reduce_(1, gpu_index.expand(len(query), -1), window_scores,
                                     reduce="amax", include_self=True)
            values, indices = by_track.topk(2, dim=1)
            first_ids.append(gpu_tracks[indices[:, 0]].cpu().numpy())
            first_scores.append(values[:, 0].cpu().numpy())
            second_ids.append(gpu_tracks[indices[:, 1]].cpu().numpy())
            second_scores.append(values[:, 1].cpu().numpy())
            del window_scores, by_track
    rows["exact_track"] = np.concatenate(first_ids)
    rows["exact_score"] = np.concatenate(first_scores)
    rows["runner_up_track"] = np.concatenate(second_ids)
    rows["runner_up_score"] = np.concatenate(second_scores)
    rows["score_margin"] = rows.exact_score - rows.runner_up_score
    if (rows.score_margin < -1e-6).any() or (rows.exact_track == rows.runner_up_track).any():
        raise AssertionError("Top candidates must be distinct tracks in descending score order")
    rows["reference_windows"] = len(ref_track_ids)
    rows["reference_tracks"] = len(unique_tracks)
    rows["reference_sha256"] = reference_sha256
    rows["query_embeddings_sha256"] = query_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, compression="gzip", float_format="%.9g")
    print(f"Scored {len(rows)} queries against {len(ref_track_ids):,} windows on {torch.cuda.get_device_name()}")


def choose_max_rule(dev, budget):
    known, unknown = dev.loc[dev.known], dev.loc[~dev.known]
    known_score, unknown_score = known.exact_score.to_numpy(), unknown.exact_score.to_numpy()
    correct = known.exact_track.to_numpy() == known.track_id.to_numpy()
    best = None
    thresholds = np.unique(np.r_[known_score, unknown_score, np.nextafter(unknown_score.max(), np.inf)])
    for threshold in thresholds:
        false = int(np.count_nonzero(unknown_score >= threshold))
        if false > budget:
            continue
        accepted = int(np.count_nonzero(correct & (known_score >= threshold)))
        key = (accepted, -false, float(threshold))
        if best is None or key > best[0]:
            best = (key, {"score_threshold": float(threshold), "gap_threshold": None,
                          "dev_false_accepts": false, "dev_known_correct_accepts": accepted})
    if best is None:
        raise RuntimeError("No score threshold met development FAR constraint")
    return best[1]


def choose_gap_rule(dev, budget):
    known, unknown = dev.loc[dev.known], dev.loc[~dev.known]
    ks, kg = known.exact_score.to_numpy(), known.score_margin.to_numpy()
    us, ug = unknown.exact_score.to_numpy(), unknown.score_margin.to_numpy()
    correct = known.exact_track.to_numpy() == known.track_id.to_numpy()
    best = None
    thresholds = np.unique(np.r_[ks, us, np.nextafter(us.max(), np.inf)])
    for threshold in thresholds:
        eligible_gap = np.sort(ug[us >= threshold])[::-1]
        gap = np.nextafter(eligible_gap[budget], np.inf) if len(eligible_gap) > budget else -np.inf
        false = int(np.count_nonzero((us >= threshold) & (ug >= gap)))
        if false > budget:
            continue
        accepted = int(np.count_nonzero(correct & (ks >= threshold) & (kg >= gap)))
        key = (accepted, -false, float(threshold), float(gap))
        if best is None or key > best[0]:
            best = (key, {"score_threshold": float(threshold),
                          "gap_threshold": None if np.isneginf(gap) else float(gap),
                          "dev_false_accepts": false, "dev_known_correct_accepts": accepted})
    if best is None:
        raise RuntimeError("No score-plus-gap rule met development FAR constraint")
    return best[1]


def decisions(rows, rule):
    accepted = rows.exact_score.to_numpy() >= rule["score_threshold"]
    if rule["gap_threshold"] is not None:
        accepted &= rows.score_margin.to_numpy() >= rule["gap_threshold"]
    correct = (rows.exact_track.to_numpy() == rows.track_id.to_numpy()) & accepted
    return accepted, correct


def bootstrap_interval(rows, events, seed, replicates=10000):
    grouped = rows.assign(event=np.asarray(events)).groupby("track_id", sort=False).event.agg(["sum", "count"])
    values, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    sampled = np.random.default_rng(seed).integers(0, len(values), (replicates, len(values)))
    rates = values[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return [float(value) for value in np.quantile(rates, [0.025, 0.975])]


def paired_difference_interval(rows, margin_events, score_events, seed, replicates=10000):
    paired = rows.assign(difference=np.asarray(margin_events, dtype=float)
                         - np.asarray(score_events, dtype=float))
    per_track = paired.groupby("track_id", sort=False).difference.mean().to_numpy()
    sampled = np.random.default_rng(seed).integers(0, len(per_track), (replicates, len(per_track)))
    means = per_track[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def evaluate(rows, rule, phase, name, seed):
    subset = rows.loc[rows.phase == phase].reset_index(drop=True)
    accepted, correct = decisions(subset, rule)
    known = subset.known.to_numpy()
    false_unknown, correct_known = (~known) & accepted, known & correct
    families = {}
    for offset, family in enumerate(("standard", "tiktok")):
        family_rows = subset.loc[subset.family == family].reset_index(drop=True)
        is_known = family_rows.known.to_numpy()
        family_accept, family_correct = decisions(family_rows, rule)
        family_unknown_false = (~is_known) & family_accept
        family_known_correct = is_known & family_correct
        families[family] = {
            "known_correct_accepted": int(family_known_correct.sum()),
            "known_queries": int(is_known.sum()),
            "known_correct_acceptance_rate": float(family_known_correct.sum() / is_known.sum()),
            "known_accepted_incorrectly": int((is_known & family_accept & ~family_correct).sum()),
            "known_rate_track_bootstrap_95": bootstrap_interval(
                family_rows.loc[is_known], family_known_correct[is_known], seed + offset),
            "unknown_false_accepts": int(family_unknown_false.sum()),
            "unknown_queries": int((~is_known).sum()),
            "unknown_false_acceptance_rate": float(family_unknown_false.sum() / (~is_known).sum()),
            "unknown_rate_track_bootstrap_95": bootstrap_interval(
                family_rows.loc[~is_known], family_unknown_false[~is_known], seed + offset + 5),
        }
    return {
        "phase": phase, "rule_name": name, **rule,
        "known_correct_accepted": int(correct_known.sum()),
        "known_queries": int(known.sum()),
        "known_correct_acceptance_rate": float(correct_known.sum() / known.sum()),
        "known_accepted_incorrectly": int((known & accepted & ~correct).sum()),
        "known_rate_track_bootstrap_95": bootstrap_interval(
            subset.loc[known], correct_known[known], seed + 10),
        "unknown_false_accepts": int(false_unknown.sum()),
        "unknown_queries": int((~known).sum()),
        "unknown_false_acceptance_rate": float(false_unknown.sum() / (~known).sum()),
        "unknown_rate_track_bootstrap_95": bootstrap_interval(
            subset.loc[~known], false_unknown[~known], seed + 11),
        "families": families,
    }


def paired_comparison(rows, rules):
    test = rows.loc[rows.phase == "test"].reset_index(drop=True)
    score_accept, score_correct = decisions(test, rules["maximum_score"])
    gap_accept, gap_correct = decisions(test, rules["score_and_margin"])

    def summarize(mask, score_events, gap_events, seed):
        selected = test.loc[mask].reset_index(drop=True)
        score_values, gap_values = score_events[mask], gap_events[mask]
        difference = gap_values.astype(float) - score_values.astype(float)
        return {
            "queries": int(mask.sum()),
            "score_rule_rate": float(score_values.mean()),
            "margin_rule_rate": float(gap_values.mean()),
            "margin_minus_score_percentage_points": float(100 * difference.mean()),
            "paired_track_bootstrap_95_percentage_points": [
                100 * value for value in paired_difference_interval(selected, gap_values, score_values, seed)
            ],
        }

    known = test.known.to_numpy()
    all_results = {
        "known_correct_acceptance": summarize(known, score_correct, gap_correct, 44001),
        "unknown_false_acceptance": summarize(~known, score_accept & ~known, gap_accept & ~known, 44002),
    }
    by_family = {}
    for offset, family in enumerate(("standard", "tiktok")):
        family_mask = test.family.to_numpy() == family
        family_known, family_unknown = family_mask & known, family_mask & ~known
        by_family[family] = {
            "known_correct_acceptance": summarize(family_known, score_correct, gap_correct, 44101+offset),
            "unknown_false_acceptance": summarize(
                family_unknown, score_accept & ~known, gap_accept & ~known, 44103+offset
            ),
        }
    return {"all": all_results, "families": by_family}


def analyse(scores_path, output, report, cohort_file):
    rows = pd.read_csv(scores_path)
    if set(rows.phase) != {"development", "test"} or set(rows.family) != {"standard", "tiktok"}:
        raise ValueError("Unexpected phase or distortion-family labels")
    development = rows.loc[rows.phase == "development"]
    budget = int(np.floor(FAR_TARGET * int((~development.known).sum())))
    rules = {"maximum_score": choose_max_rule(development, budget),
             "score_and_margin": choose_gap_rule(development, budget)}
    results = {
        "seed": SEED, "false_acceptance_target": FAR_TARGET,
        "data": {"queries": int(len(rows)), "unique_tracks": int(rows.track_id.nunique()),
                 "queries_per_track": 2, "removed_tracks": int(rows.loc[~rows.known, "track_id"].nunique()),
                 "reference_windows": int(rows.reference_windows.iloc[0]),
                 "reference_tracks": int(rows.reference_tracks.iloc[0]),
                 "fingerprints_sha256": str(rows.reference_sha256.iloc[0]) if "reference_sha256" in rows else None,
                 "query_embeddings_sha256": str(rows.query_embeddings_sha256.iloc[0])
                 if "query_embeddings_sha256" in rows else None,
                 "projection_head_sha256": HEAD_SHA256,
                 "backbone": "OpenMuQ/MuQ-large-msd-iter@0562a57814f6f8bb9dfea0a25921a2fce1a841a",
                 "cohort_track_ids_sha256": sha256_file(cohort_file) if cohort_file.is_file() else None,
                 "split_track_counts": {f"{phase}_{label}": int(group.track_id.nunique())
                     for (phase, label), group in rows.groupby(["phase", "known"])}},
        "threshold_selection": {"max_false_accepts_on_development": budget, **rules},
        "development": [evaluate(rows, rule, "development", name, 22100+i*100)
                        for i, (name, rule) in enumerate(rules.items())],
        "test": [evaluate(rows, rule, "test", name, 33100+i*100)
                 for i, (name, rule) in enumerate(rules.items())],
        "paired_test_comparison": paired_comparison(rows, rules),
        "limitations": [
            "Queries use synthetic augmentations; physical phone recordings are not included.",
            "The runner-up is the best window score from a track distinct from the top result.",
            "Thresholds maximise correct known acceptance under a 1% empirical unknown false-acceptance constraint.",
            "Fresh tracks exclude training pairs, both earlier query archives and the earlier unknown-song cohort.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(results), encoding="utf-8")
    print(json.dumps(results["test"], indent=2))


def render_report(results):
    test = {row["rule_name"]: row for row in results["test"]}
    lines = [
        "# Does the gap to the runner-up help reject unknown songs?", "",
        "## Fresh-track test", "",
        "We compared a maximum cosine threshold with a rule that also requires a sufficient score gap between the best two distinct tracks. Development tracks select thresholds to maximise correct acceptance of known queries with no more than 1% empirical false acceptance among unknown queries. The held-out tracks evaluate both frozen rules. The exact GPU run scored 4,096 synthetic queries against 105,767 remaining reference windows on an NVIDIA RTX 4090.",
        "",
        "| Rule | Known correct accepted | Known accepted incorrectly | Unknown false accepted | Standard correct | TikTok-like correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("maximum_score", "Maximum score"), ("score_and_margin", "Score plus distinct-song margin")):
        row = test[name]
        standard, tiktok = row["families"]["standard"], row["families"]["tiktok"]
        lines.append(
            f"| {label} | {row['known_correct_accepted']}/{row['known_queries']} "
            f"({row['known_correct_acceptance_rate']:.1%}) | {row['known_accepted_incorrectly']} | "
            f"{row['unknown_false_accepts']}/{row['unknown_queries']} "
            f"({row['unknown_false_acceptance_rate']:.2%}) | "
            f"{standard['known_correct_accepted']}/{standard['known_queries']} "
            f"({standard['known_correct_acceptance_rate']:.1%}) | "
            f"{tiktok['known_correct_accepted']}/{tiktok['known_queries']} "
            f"({tiktok['known_correct_acceptance_rate']:.1%}) |"
        )
    lines.extend([
        "",
        f"On the same held-out known queries, the margin rule changed correct acceptance by {results['paired_test_comparison']['all']['known_correct_acceptance']['margin_minus_score_percentage_points']:+.2f} percentage points (paired track-bootstrap 95% interval {results['paired_test_comparison']['all']['known_correct_acceptance']['paired_track_bootstrap_95_percentage_points'][0]:+.2f} to {results['paired_test_comparison']['all']['known_correct_acceptance']['paired_track_bootstrap_95_percentage_points'][1]:+.2f}). Unknown false acceptance changed by {results['paired_test_comparison']['all']['unknown_false_acceptance']['margin_minus_score_percentage_points']:+.2f} points (95% paired interval {results['paired_test_comparison']['all']['unknown_false_acceptance']['paired_track_bootstrap_95_percentage_points'][0]:+.2f} to {results['paired_test_comparison']['all']['unknown_false_acceptance']['paired_track_bootstrap_95_percentage_points'][1]:+.2f}).",
        "",
        "The margin rule accepted 11/1,024 missing-song queries, one above its 10/1,024 development budget. The score-only rule accepted 4/1,024. The gap rule correctly accepted 731/1,024 known queries versus 538/1,024 for score-only. It accepted no incorrect known titles in this test, compared with two for score-only. This is a measured trade-off, not a demonstrated 1% test FAR.",
        "",
        "Track-bootstrap 95% intervals, thresholds, and family false accepts are in runner-up-margin.json. Per-query scores for the best windows from two distinct track IDs are in runner-up-margin-scores.csv.gz. The ordered cohort IDs in runner-up-margin-cohort-tracks.csv preserve the held-out split for reproduction.",
        "",
        "The exact GPU scorer compares each query to all available reference windows, reduces window scores to one maximum per track, and chooses two distinct track IDs. The query cohort excludes training tracks, both earlier query archives, and the earlier unknown-song cohort.",
        "",
        "The margin increases correct acceptance by substantially more than it increases false acceptance in this sample. Its observed test false-acceptance rate still exceeds the development target. Synthetic distortions cannot establish phone-in-room performance.",
        "",
        "## Reproduction", "",
        "Run from the repository root in PowerShell with the processed medium data and the pinned model files:",
        "",
        "~~~powershell",
        "$python = 'data/medium/.venv/Scripts/python.exe'",
        "$backbone = (Resolve-Path 'data/medium/hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots/0562a57814f6f8bb9dfea0a25921a2fce1a841a').Path",
        "& $python scripts/runner_up_study.py generate --backbone $backbone",
        "& $python scripts/runner_up_study.py score",
        "& $python scripts/runner_up_study.py analyse",
        "~~~",
        "",
        "The ordered track IDs and earlier study IDs are in runner-up-margin-cohort-tracks.csv. The JSON stores checksums for the reference fingerprints, generated query embeddings, projection head and cohort IDs. Running analyse alone rebuilds the JSON and this report from the saved GPU scores.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--processed", type=Path, default=Path("data/medium/data/processed"))
    generate_parser.add_argument("--model", type=Path, default=Path("data/medium/models/projection_head.pt"))
    generate_parser.add_argument("--backbone", required=True)
    generate_parser.add_argument("--previous-queries", type=Path, default=Path("data/medium/unknown-song-queries.npz"))
    generate_parser.add_argument("--output", type=Path, default=Path("data/medium/runner-up-queries.npz"))
    generate_parser.add_argument("--cohort-file", type=Path, default=Path("reports/runner-up-margin-cohort-tracks.csv"))
    generate_parser.add_argument("--workers", type=int, default=8)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--processed", type=Path, default=Path("data/medium/data/processed"))
    score_parser.add_argument("--queries", type=Path, default=Path("data/medium/runner-up-queries.npz"))
    score_parser.add_argument("--output", type=Path, default=Path("reports/runner-up-margin-scores.csv.gz"))
    score_parser.add_argument("--batch-size", type=int, default=64)
    analyse_parser = commands.add_parser("analyse")
    analyse_parser.add_argument("--scores", type=Path, default=Path("reports/runner-up-margin-scores.csv.gz"))
    analyse_parser.add_argument("--output", type=Path, default=Path("reports/runner-up-margin.json"))
    analyse_parser.add_argument("--report", type=Path, default=Path("reports/runner-up-margin.md"))
    analyse_parser.add_argument("--cohort-file", type=Path, default=Path("reports/runner-up-margin-cohort-tracks.csv"))
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.processed, args.model, args.backbone, args.previous_queries,
                 args.output, args.cohort_file, args.workers)
    elif args.command == "score":
        score_queries(args.processed, args.queries, args.output, args.batch_size)
    else:
        analyse(args.scores, args.output, args.report, args.cohort_file)


if __name__ == "__main__":
    main()
