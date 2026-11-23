"""Resolve the exact earlier benchmark used as a dependency of this independent PR."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "data/large/fresh"
PIN = "b3d6a10c8efa4b653a5d90c2bade38f052400d87"


def blob(name):
    """Read a baseline file from the pinned object without switching branches."""
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{name}"])


def support():
    """Load the original verifier, calibrator and LSH kernel from pinned PR 9."""
    path = FRESH / "acoustic" / "baseline" / PIN
    path.mkdir(parents=True, exist_ok=True)
    for file in ("landmark_verify.py", "landmark_evaluation.py", "fresh_shortlist.py", "landmark_latency.py"):
        local = path / file
        if not local.exists():
            local.write_bytes(blob("scripts/" + file))
    os.environ.setdefault("HF_HOME", str(ROOT / "data/medium/hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(path))
    sys.path.insert(0, str(ROOT / "src"))


def previous_scores():
    """Get the per-query baseline for a byte-identical paired comparison."""
    with np.load(io.BytesIO(blob("reports/landmark-scores.npz"))) as d:
        return {k: d[k] for k in d}


def previous_results():
    """Read the previously frozen development rules and held-out totals."""
    return json.loads(blob("reports/landmark-results.json"))


def groups():
    """Read duplicate-group labels and the full set of absent groups."""
    import csv

    with (FRESH / "groups.csv").open() as source:
        mapping = {int(r["track_id"]): int(r["group_id"]) for r in csv.DictReader(source)}
    with (FRESH / "cohort.csv").open() as source:
        absent = {int(r["group_id"]) for r in csv.DictReader(source) if not int(r["known"])}
    return mapping, absent
