"""Export one real LSH table and three Medium searches for the public lesson."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from afp.lsh import AngularLSH

PROCESSED = ROOT / "data/medium/data/processed"
OUTPUT = ROOT / "explorer/public/ruler.json"


def main():
    with (PROCESSED / "manifest.csv").open(encoding="utf-8", newline="") as file:
        labels = {int(r["track_id"]): [r["title"], r["artist"]] for r in csv.DictReader(file)}

    with np.load(PROCESSED / "fingerprints/references.npz") as refs:
        index = AngularLSH(refs["vectors"], refs["track_id"], bits=10, tables=64, seed=2807)
    tracks = index.track_ids
    order, offsets = index.buckets[0]
    buckets = []
    used = set()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        rows = order[start:end]
        # Original row order supplies a reproducible sample; songs may have several clips.
        sample = list(dict.fromkeys(int(t) for t in tracks[rows]))[:3]
        buckets.append(sample)
        used.update(sample)

    cases = []
    with np.load(PROCESSED / "fingerprints/queries.npz") as archived:
        ids, vectors = archived["track_id"], archived["vectors"]
        for row in (2, 5, 6):
            query = vectors[row]
            projections = (query @ index.planes).reshape(index.tables, index.bits)
            keys = (projections >= 0) @ index.weights
            uncertain = np.argsort(abs(projections), axis=1, kind="stable")[:, :2]
            windows = []
            for key, (a, b), (references, boundaries) in zip(keys, uncertain, index.buckets, strict=True):
                probes = [int(key ^ mask) for mask in (0, 1 << a, 1 << b, (1 << a) | (1 << b))]
                if not windows:
                    first_table = probes
                windows.extend(references[boundaries[k]:boundaries[k + 1]] for k in probes)
            candidates = np.unique(np.concatenate(windows))
            scores = index.references[candidates] @ query
            winner = int(tracks[candidates[np.argmax(scores)]])
            other = scores[tracks[candidates] != winner]
            score = float(scores.max())
            gap = score - float(other.max())
            used.update((winner, int(ids[row])))
            cases.append(dict(row=row, source=int(ids[row]), start=float(archived["start_s"][row]),
                              probes=first_table, candidates=len(candidates), winner=winner,
                              accepted=score >= .749896646 and gap >= .117484212))
    counts = np.diff(offsets).tolist()
    assert len(counts) == 1024 and sum(counts) == len(tracks)
    assert all(len(set(case["probes"])) == 4 for case in cases)
    payload = dict(counts=counts, examples=buckets, labels={i: labels[i] for i in sorted(used)},
                   windows=len(tracks), tracks=len(np.unique(tracks)), cases=cases)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
