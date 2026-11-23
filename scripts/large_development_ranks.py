"""Measure how often the correct recording reaches an exact cosine shortlist on Large.

Run from the repository root with the medium pipeline's CUDA PyTorch environment.
Only development query vectors and labels are scored. Missing IDs from both phases
define the same searchable catalogue as the rejection experiment.
"""

import json
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def main():
    torch.set_float32_matmul_precision("highest")
    with np.load("data/medium/runner-up-queries.npz") as queries:
        absent_ids = np.unique(queries["track_id"][~queries["known"]])
        select = (queries["phase"] == "development") & queries["known"]
        tracks, vectors, families, distortions = (
            queries[key][select] for key in ("track_id", "vectors", "family", "distortion")
        )
    with np.load("data/large/references.npz") as refs:
        keep = ~np.isin(refs["track_id"], absent_ids)
        ref_tracks = refs["track_id"][keep]
        refs_gpu = torch.from_numpy(refs["vectors"][keep]).cuda()
    ids, compact = np.unique(ref_tracks, return_inverse=True)
    truth = np.searchsorted(ids, tracks)
    np.testing.assert_array_equal(ids[truth], tracks)
    compact_gpu = torch.as_tensor(compact, dtype=torch.long, device="cuda")
    ranks = []
    for start in range(0, len(tracks), 32):
        batch = torch.from_numpy(vectors[start : start + 32]).cuda()
        scores = batch @ refs_gpu.T
        winners = compact_gpu[scores.argmax(dim=1)]
        by_track = scores.new_full((len(batch), len(ids)), -torch.inf)
        by_track.scatter_reduce_(1, compact_gpu.expand(len(batch), -1), scores, reduce="amax")
        target = torch.as_tensor(truth[start : start + len(batch)], device="cuda")
        target_score = by_track.gather(1, target[:, None])
        # Count tied tracks ahead for the conservative top-k bound.
        worst_rank = (by_track >= target_score).sum(dim=1)
        best_rank = 1 + (by_track > target_score).sum(dim=1)
        ranks.append(torch.stack([winners == target, best_rank, worst_rank], dim=1).cpu().numpy())
    top1, best, worst = np.concatenate(ranks).T

    def counts(mask):
        return {
            "queries": int(mask.sum()),
            "strict_top1": int(top1[mask].sum()),
            "top5_even_if_ties_lose": int((worst[mask] <= 5).sum()),
            "top5_if_ties_win": int((best[mask] <= 5).sum()),
            "top5_recoverable_top1_misses": int(((top1[mask] == 0) & (worst[mask] <= 5)).sum()),
            "outside_top5_even_if_ties_win": int((best[mask] > 5).sum()),
        }

    groups = {f: counts(families == f) for f in np.unique(families)}
    conditions = {d: counts(distortions == d) for d in np.unique(distortions)}
    result = {
        "split": "Large open-set catalogue, known development queries only",
        "gpu": torch.cuda.get_device_name(),
        "windows": len(ref_tracks),
        "searchable_tracks": len(ids),
        "known_songs": len(np.unique(tracks)),
        "by_family": groups,
        "by_distortion": conditions,
    }
    Path("reports/large-development-ranks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
