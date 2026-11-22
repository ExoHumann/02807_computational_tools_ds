"""Track identification with time-consistent votes from several fingerprint windows."""

import torch


def match_windows(
    scores: torch.Tensor,
    candidate_tracks: torch.Tensor,
    reference_starts: torch.Tensor,
    query_starts: torch.Tensor,
    tolerance_s: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strongest-window track, time-vote track and number of supporting query windows.

    Inputs are [batch, windows, neighbours] with descending cosine scores within each window.
    Each candidate supplies a (track, reference time minus query time) hypothesis. Other
    windows contribute their best positive cosine score for this track within the tolerance.
    Duplicate candidates within a query window cannot increase its vote. Score ties use
    the first hypothesis, in query-time then neighbour order.
    """
    if scores.ndim != 3 or scores.shape != candidate_tracks.shape or scores.shape != reference_starts.shape:
        raise ValueError("Scores, track ids and reference times must have matching [batch, windows, neighbours] shapes")
    batch, windows, neighbours = scores.shape
    if windows == 0 or neighbours == 0 or query_starts.shape != (windows,) or tolerance_s < 0:
        raise ValueError("Expected nonempty candidate windows, matching query times and a nonnegative tolerance")
    tracks = candidate_tracks.reshape(batch, -1)
    offsets = reference_starts - query_starts[None, :, None]
    hypothesis_time = offsets.reshape(batch, -1)
    same_song = tracks[:, :, None, None] == candidate_tracks[:, None, :, :]
    aligned = (hypothesis_time[:, :, None, None] - offsets[:, None, :, :]).abs() <= tolerance_s
    matches = torch.where(same_song & aligned, scores[:, None, :, :], 0.0).clamp_min(0).amax(dim=-1)
    best = matches.sum(dim=-1).argmax(dim=1)
    row = torch.arange(batch, device=scores.device)
    strongest = scores.reshape(batch, -1).argmax(dim=1)
    return tracks[row, strongest], tracks[row, best], (matches[row, best] > 0).sum(dim=1)
