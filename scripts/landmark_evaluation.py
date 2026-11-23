"""Fit decision rules on development songs and replay a paired recognition and rejection benchmark."""

import json
from pathlib import Path

import numpy as np

REPORT = Path("reports")
TARGETS = (0.0025, 0.01, 0.02, 0.05, 0.10)
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def evidence(data, weight):
    """Fuse candidate scores, then measure winner confidence and separation."""
    fused = data["landmarks"] if weight == "landmarks" else data["scores"] + weight * data["landmarks"]
    order = np.argsort(-fused, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(fused, order, axis=1).astype("f8")
    candidates = np.take_along_axis(data["candidate_groups"], order, axis=1)
    correct = (candidates[:, 0] == data["group_id"]) & data["known"].astype(bool)
    return sorted_scores[:, 0], sorted_scores[:, 0] - sorted_scores[:, 1], correct


def fit(score, gap, correct, known, budget):
    """Maximise accepted correct titles subject to the development unknown budget."""
    best = None
    for cut in np.unique(np.r_[0.0, gap[correct]]):
        eligible = gap >= cut
        unknown_scores = score[eligible & ~known]
        threshold = -1.0e6
        if len(unknown_scores) > budget:
            critical = np.partition(unknown_scores, -budget - 1)[-budget - 1]
            threshold = float(np.nextafter(critical, np.inf))
        accept = eligible & (score >= threshold)
        priority = (
            int((correct & accept).sum()),
            -int((~known & accept).sum()),
            -int((known & ~correct & accept).sum()),
            -cut,
        )
        if best is None or priority > best[0]:
            best = (priority, float(cut), threshold)
    return best


def outcome(score, gap, correct, known, rule):
    """Count both kinds of mistakes, correct answers and abstentions."""
    accept = (score >= rule["score"]) & (gap >= rule["gap"])
    return (
        {
            "correct": int((correct & accept).sum()),
            "wrong_known": int((known & ~correct & accept).sum()),
            "false_accepts": int((~known & accept).sum()),
            "known": int(known.sum()),
            "unknown": int((~known).sum()),
            "closed_top1": int(correct.sum()),
        },
        correct & accept,
        (~known) & accept,
    )


def ci(delta, groups, mask):
    """Resample recording families so both distorted clips remain together."""
    ids, indices = np.unique(groups[mask], return_inverse=True)
    successes = np.bincount(indices, weights=delta[mask])
    counts = np.bincount(indices)
    rng = np.random.default_rng(28079)
    samples = rng.integers(0, len(ids), size=(5000, len(ids)))
    rates = successes[samples].sum(1) / counts[samples].sum(1)
    return np.percentile(rates * 100, [2.5, 97.5]).tolist()


def evaluate(data):
    """Freeze all operating points using development data before computing test outcomes."""
    dev = data["phase"] == "development"
    test = data["phase"] == "test"
    known = data["known"].astype(bool)
    assert set(data["group_id"][dev]).isdisjoint(data["group_id"][test])
    assert all(len(set(row)) == 5 for row in data["candidate_groups"])
    assert not set(data["group_id"][~known]).intersection(data["candidate_groups"].ravel())
    evidence_by_weight = {w: evidence(data, w) for w in (*WEIGHTS, "landmarks")}
    runs = []
    for target in TARGETS:
        budget = int(target * (dev & ~known).sum())
        fits = {}
        for weight, (score, gap, correct) in evidence_by_weight.items():
            fits[weight] = fit(score[dev], gap[dev], correct[dev], known[dev], budget)
        best_weight = max(WEIGHTS, key=lambda w: fits[w][0][:3])
        for method, weight in (("cosine", 0.0), ("landmarks_only", "landmarks"), ("fusion", best_weight)):
            _, g, s = fits[weight]
            rule = {"score": s, "gap": g, "landmark_weight": weight}
            runs.append({"method": method, "target": target, "rule": rule})
    # With abstention disabled choose the fusion weight by development rank-one accuracy.
    closed_weight = max(WEIGHTS, key=lambda w: evidence_by_weight[w][2][dev].sum())
    for method, w in (("cosine", 0.0), ("landmarks_only", "landmarks"), ("fusion", closed_weight)):
        runs.append({"method": method, "target": None, "rule": {"score": -1.0e6, "gap": 0.0, "landmark_weight": w}})
    paired = {}
    for run in runs:
        score, gap, correct = evidence_by_weight[run["rule"]["landmark_weight"]]
        rule = run["rule"]
        for key, mask in (("dev", dev), ("test", test)):
            run[key], _, _ = outcome(score[mask], gap[mask], correct[mask], known[mask], rule)
        if run["target"] == 0.01:
            _, ok, fa = outcome(score, gap, correct, known, rule)
            run["unknown_rate_ci_pct"] = ci(fa, data["group_id"], test & ~known)
            paired[run["method"]] = ok
            weight = rule["landmark_weight"]
            votes = data["landmarks"] if weight == "landmarks" else data["scores"] + weight * data["landmarks"]
            selected = votes.argmax(axis=1)
            strict = data["candidates"][np.arange(len(test)), selected] == data["track_id"]
            run["strict_id_correct_accepted"] = int((ok & strict & test).sum())
            run["distortions"] = {}
            for distortion in sorted(set(data["distortion"])):
                mask = test & (data["distortion"] == distortion)
                run["distortions"][str(distortion)] = outcome(score[mask], gap[mask], correct[mask], known[mask], rule)[
                    0
                ]
            run["families"] = {}
            for f in sorted(set(data["family"])):
                mask = test & (data["family"] == f)
                run["families"][str(f)] = outcome(score[mask], gap[mask], correct[mask], known[mask], rule)[0]
    old, new = paired["cosine"], paired["fusion"]
    changes = {
        "retained": int((old & new & test).sum()),
        "lost": int((old & ~new & test).sum()),
        "gained": int((~old & new & test).sum()),
        "paired_change_pp_ci": ci(new.astype(int) - old, data["group_id"], test & known),
    }
    baseline_winner = evidence_by_weight[0.0][2]
    changes["gained_by_promoting_candidate"] = int((test & new & ~old & ~baseline_winner).sum())
    changes["gained_by_accepting_previous_winner"] = int((test & new & ~old & baseline_winner).sum())
    closed_winner = evidence_by_weight[closed_weight][2]
    changes["closed_promoted"] = int((test & closed_winner & ~baseline_winner).sum())
    changes["closed_demoted"] = int((test & baseline_winner & ~closed_winner).sum())
    changes["paired_closed_change_pp_ci"] = ci(
        closed_winner.astype(int) - baseline_winner, data["group_id"], test & known
    )
    ceiling = np.any(data["candidate_groups"] == data["group_id"][:, None], axis=1)
    return {
        "runs": runs,
        "paired": changes,
        "top5": int((ceiling & test & known).sum()),
        "top5_by_family": {
            str(f): int((ceiling & test & known & (data["family"] == f)).sum()) for f in set(data["family"])
        },
    }


def main():
    """Publish the rule choices and exact counts used by the notebook and report."""
    with np.load(REPORT / "landmark-scores.npz", allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved}
    result = evaluate(data)
    (REPORT / "landmark-results.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "paired": result["paired"],
                "top5": result["top5"],
                "points": [r for r in result["runs"] if r["target"] in (0.01, None)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
