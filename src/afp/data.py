"""Data pipeline: find tracks, decode audio to 24 kHz mono and cut it into fixed-length segments.

Only :func:`segment` is implemented. Decoding and track discovery are stubs until the dataset is chosen.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 24_000
SEGMENT_S = 10.0
HOP_S = 5.0


def list_tracks(root: Path) -> pd.DataFrame:
    """List the audio files below ``root``.

    Args:
        root: Directory holding the raw audio, e.g. ``data/raw``.

    Returns:
        A table with at least a ``track_id`` and a ``path`` column, one row per track.
    """
    raise NotImplementedError


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to a mono ``float32`` waveform at ``sample_rate``.

    Args:
        path: Audio file to decode.
        sample_rate: Target sample rate in Hz. 24 kHz is what MuQ and MERT expect.

    Returns:
        A 1D ``float32`` array.
    """
    raise NotImplementedError


def segment(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segment_s: float = SEGMENT_S,
    hop_s: float = HOP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a waveform into fixed-length, possibly overlapping windows.

    Trailing audio shorter than one window is dropped, and a waveform shorter than one window yields no segments.

    Args:
        waveform: 1D waveform.
        sample_rate: Sample rate of ``waveform`` in Hz.
        segment_s: Window length in seconds.
        hop_s: Distance between window starts in seconds.

    Returns:
        A tuple ``(segments, start_s)``: ``segments`` has shape ``[n, window_samples]`` and ``start_s`` holds the
        start time of each window in seconds.

    Raises:
        ValueError: If ``waveform`` is not 1D or the window or hop is shorter than one sample.
    """
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1D, got shape {waveform.shape}")
    window = round(segment_s * sample_rate)
    hop = round(hop_s * sample_rate)
    if window < 1 or hop < 1:
        raise ValueError("segment_s and hop_s must each cover at least one sample")
    if len(waveform) < window:
        return np.empty((0, window), dtype=waveform.dtype), np.empty(0, dtype=np.float64)
    segments = np.lib.stride_tricks.sliding_window_view(waveform, window)[::hop].copy()
    start_s = np.arange(len(segments)) * hop / sample_rate
    return segments, start_s
