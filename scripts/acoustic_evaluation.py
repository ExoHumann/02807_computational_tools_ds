"""Choose operating points on development songs and score the previously identified LSH misses."""

import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("lsh5", "lsh10", "dual")
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TARGETS = (0.0025, 0.01, 0.02, 0.05, 0.10)
REPORTS = Path("reports")


def inputs(d, method):
    """Select the same frozen neural shortlist or one of the expanded pools."""
    tag = "lsh10" if method == "lsh5" else method
    width = 5 if method == "lsh5" else 10
    return tuple(d[f"{tag}_{k}"][:, :width] for k in ("groups", "cos", "lm"))


def ranks(d, method, weight):
    """Rank valid groups using winner score, with stable ties favoring original order."""
    group, cos, lm = inputs(d, method)
    score = cos + weight * lm
    score = np.where(group >= 0, score, -1e6)
    order = np.argsort(-score, axis=1, kind="stable")
    groups = np.take_along_axis(group, order, axis=1)
    ordered = np.take_along_axis(score, order, axis=1).astype("f8")
    return groups, ordered[:, 0], ordered[:, 0] - ordered[:, 1]


def decisions(d, method, rule):
    """Replay a locked ranking and open-set decision with no model or GPU."""
    g, s, gap = ranks(d, method, rule["weight"])
    correct = (g[:, 0] == d["group_id"]) & d["known"]
    accept = (s >= rule["score"]) & (gap >= rule["gap"])
    return g, correct, accept


def counts(d, method, rule, mask):
    """Return listener outcomes and coverage at fixed five and full pool sizes."""
    g, correct, answer = decisions(d, method, rule)
    known = d["known"] & mask
    unknown = ~d["known"] & mask
    same = g == d["group_id"][:, None]
    return {
        "known": int(known.sum()),
        "unknown": int(unknown.sum()),
        "top1": int((known & correct).sum()),
        "top5": int((known & same[:, :5].any(1)).sum()),
        "pool": int((known & same.any(1)).sum()),
        "correct": int((known & answer & correct).sum()),
        "wrong_known": int((known & answer & ~correct).sum()),
        "fa": int((unknown & answer).sum()),
    }


def evaluate(d):
    """Fit on development and record all outcomes without revising the nominated tracks."""
    from acoustic_baseline import previous_results, support

    support()
    from landmark_evaluation import ci, fit

    dev = d["phase"] == "development"
    test = d["phase"] == "test"
    known = d["known"]
    assert not set(d["group_id"][dev]) & set(d["group_id"][test])
    assert all(len(set(row[row >= 0])) == sum(row >= 0) for row in d["dual_groups"])
    assert not set(d["group_id"][~known]) & set(d["dual_groups"].ravel())
    frozen = {}
    for method in METHODS:
        proposals = {}
        for w in WEIGHTS:
            g, s, gap = ranks(d, method, w)
            ok = (g[:, 0] == d["group_id"]) & known
            for target in TARGETS:
                priority, gcut, scut = fit(s[dev], gap[dev], ok[dev], known[dev], int(target * (dev & ~known).sum()))
                rule = {"weight": w, "score": scut, "gap": gcut}
                if target not in proposals or priority[:3] > proposals[target][0]:
                    proposals[target] = (priority[:3], rule)
            value = int((ok & dev).sum())
            if "closed" not in proposals or value > proposals["closed"][0]:
                proposals["closed"] = (value, {"weight": w, "score": -1e6, "gap": 0.0})
        frozen[method] = {str(k): v for k, (_, v) in proposals.items()}

    results = {}
    for m in METHODS:
        results[m] = {}
        for tag, rule in frozen[m].items():
            results[m][tag] = {"rule": rule, "development": counts(d, m, rule, dev), "test": counts(d, m, rule, test)}
    for baseline in previous_results()["runs"]:
        if baseline["method"] == "fusion":
            tag = "closed" if baseline["target"] is None else str(baseline["target"])
            if tag in results["lsh5"]:
                replay = results["lsh5"][tag]["test"]
                assert replay["correct"] == baseline["test"]["correct"]
                assert replay["fa"] == baseline["test"]["false_accepts"]
    same_rule = {tag: counts(d, "dual", rule, test) for tag, rule in frozen["lsh5"].items()}
    baseline_pool, _, _ = inputs(d, "lsh5")
    missed = test & known & ~(baseline_pool == d["group_id"][:, None]).any(1)
    assert missed.sum() == 22
    rows = []
    for i in np.flatnonzero(missed):
        row = {
            "query": int(i),
            "track_id": int(d["track_id"][i]),
            "group_id": int(d["group_id"][i]),
            "distortion": str(d["distortion"][i]),
        }
        for m in ("lsh10", "dual"):
            g, correct, _ = decisions(d, m, frozen[m]["closed"])
            match = np.flatnonzero(g[i] == d["group_id"][i])
            row[m + "_rank"] = int(match[0] + 1) if len(match) else 0
            _, correct, accept = decisions(d, m, frozen[m]["0.01"])
            row[m + "_correct_accepted"] = bool(correct[i] & accept[i])
        rows.append(row)
    paired = {}
    for tag in ("closed", "0.01"):
        _, bc, ba = decisions(d, "lsh5", frozen["lsh5"][tag])
        _, dc, da = decisions(d, "dual", frozen["dual"][tag])
        before = bc & ba
        after = dc & da
        paired[tag] = {
            "gained": int((test & after & ~before).sum()),
            "lost": int((test & before & ~after).sum()),
            "correct_delta_95_pp": ci(after.astype(int) - before, d["group_id"], test & known),
        }
        if tag == "0.01":
            paired[tag]["fa_delta_95_pp"] = ci(da.astype(int) - ba, d["group_id"], test & ~known)
            paired[tag]["dual_fa_95_pct"] = ci(da, d["group_id"], test & ~known)
    return {
        "methods": results,
        "previous_rule_on_dual": same_rule,
        "paired": paired,
        "misses": rows,
        "acoustic_only_top5": int((test & known & (d["acoustic_groups"] == d["group_id"][:, None]).any(1)).sum()),
        "mean_dual_pool": float((d["dual_groups"][test] >= 0).sum(1).mean()),
    }


def main():
    """Save the matched evaluation and one auditable row per historical miss."""
    with np.load(REPORTS / "acoustic-nominations.npz") as archive:
        d = {k: archive[k] for k in archive}
    result = evaluate(d)
    for tag in ("closed", "0.01"):
        print(tag, {m: result["methods"][m][tag]["test"] for m in METHODS})
    with (REPORTS / "acoustic-original-misses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result["misses"][0].keys())
        w.writeheader()
        w.writerows(result["misses"])
    (REPORTS / "acoustic-results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
