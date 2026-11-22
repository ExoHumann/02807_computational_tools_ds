"""Regression checks for temporal matching, independent of audio and the encoder."""

import pytest
import torch

from afp.temporal import match_windows


def test_consistent_offsets_overrule_one_lucky_match():
    scores = torch.tensor([[[0.91, 0.83], [0.84, 0.79], [0.86, 0.81]]])
    tracks = torch.tensor([[[8, 7], [7, 8], [7, 8]]])
    times = torch.tensor([[[20.0, 5.0], [9.0, 45.0], [13.0, 60.0]]])
    strongest, voted, support = match_windows(scores, tracks, times, torch.tensor([0.0, 4.0, 8.0]), 0.1)
    assert (strongest.item(), voted.item(), support.item()) == (8, 7, 3)


def test_one_window_cannot_vote_twice_and_ties_are_stable():
    scores = torch.tensor([[[0.9, 0.9], [0.8, 0.7]]])
    tracks = torch.tensor([[[11, 11], [5, 5]]])
    times = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    _, voted, support = match_windows(scores, tracks, times, torch.tensor([0.0, 4.0]))
    assert voted.item() == 11 and support.item() == 1
    with pytest.raises(ValueError):
        match_windows(scores, tracks, times, torch.tensor([0.0]))
