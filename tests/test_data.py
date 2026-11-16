import numpy as np

from afp.data import SAMPLE_RATE, segment


def test_segment_30s_clip_gives_five_overlapping_windows():
    waveform = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
    segments, start_s = segment(waveform)
    assert segments.shape == (5, 10 * SAMPLE_RATE)
    np.testing.assert_allclose(start_s, [0.0, 5.0, 10.0, 15.0, 20.0])
