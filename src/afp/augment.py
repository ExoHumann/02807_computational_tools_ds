"""Audio augmentations used to build query/reference pairs, following the paper (Section 2.3).

Nothing is implemented yet. ``PARAMETER_RANGES`` records the ranges the paper samples from.
"""

import numpy as np

AUGMENTATIONS = (
    "time_stretch",
    "pitch_shift",
    "noise",
    "reverb",
    "band_pass",
    "high_pass",
    "low_pass",
    "echo",
    "encodec",
)

PARAMETER_RANGES = {
    "time_stretch": (0.7, 1.5),
    "pitch_shift": (-5.0, 5.0),
    "band_pass": (300.0, 1800.0),
    "high_pass": (1800.0, 3400.0),
    "low_pass": (300.0, 1500.0),
    "echo": (0.1, 0.2),
    "encodec": (6, 12),
}


def augment(waveform: np.ndarray, name: str, rng: np.random.Generator, sample_rate: int) -> np.ndarray:
    """Apply one named augmentation with parameters drawn from ``PARAMETER_RANGES``.

    Ranges are: time stretch factor, pitch shift in semitones, filter cut-offs in Hz (band-pass is the pass band,
    high- and low-pass the cut-off), echo delay in seconds, and Encodec bit-width.

    Args:
        waveform: 1D waveform.
        name: One of ``AUGMENTATIONS``.
        rng: Random generator used to draw the parameters.
        sample_rate: Sample rate of ``waveform`` in Hz.

    Returns:
        The augmented 1D waveform.
    """
    raise NotImplementedError
