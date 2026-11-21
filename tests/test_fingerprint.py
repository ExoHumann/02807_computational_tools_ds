import numpy as np
import pandas as pd
import torch

from afp.augment import AUGMENTATIONS, _pitch_shift, augment
from afp.data import SAMPLE_RATE, track_path
from afp.fingerprint import cache_clean, cache_clips, encode, split_tracks, train


class FakeBackbone:
    """Stand-in for MuQ: 32 log-spectrum band energies per window."""

    def encode(self, waveforms: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(waveforms, axis=1))[:, :120_000]
        return np.log1p(spectrum.reshape(len(waveforms), 32, -1).mean(axis=2)).astype(np.float32)


def test_augmentations_are_finite_and_only_time_stretch_changes_length():
    rng = np.random.default_rng(0)
    waveform = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    for name in AUGMENTATIONS:
        out = augment(waveform, name, rng, SAMPLE_RATE)
        assert np.isfinite(out).all() and np.abs(out).max() <= 1.0, name
        assert (len(out) == len(waveform)) == (name != "time_stretch"), name
    shifted = _pitch_shift(torch.from_numpy(waveform), 12.0).numpy()
    peak_hz = np.fft.rfftfreq(len(shifted), 1 / SAMPLE_RATE)[np.abs(np.fft.rfft(shifted)).argmax()]
    assert abs(peak_hz - 880.0) < 2.0


def test_fingerprint_pipeline_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for track_id in range(6):
        t = np.arange(25 * SAMPLE_RATE) / SAMPLE_RATE
        tone = sum(np.sin(2 * np.pi * f * t) for f in rng.uniform(100, 4000, size=3)) / 3
        path = track_path(tmp_path, track_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, (tone * 20_000).astype(np.int16))
        rows.append({"track_id": track_id, "path": path.relative_to(tmp_path).as_posix()})
    manifest = pd.DataFrame(rows)
    train_ids, eval_ids = split_tracks(manifest["track_id"], 0.5, seed=0)
    assert set(train_ids).isdisjoint(eval_ids)

    backbone = FakeBackbone()
    cache_clean(manifest, tmp_path, backbone, batch_size=4, tracks_per_shard=3)
    cache_clips("pairs", manifest, train_ids, 12, tmp_path, backbone, True, seed=0, batch_size=4, clips_per_shard=6)
    cache_clips("queries", manifest, eval_ids, 6, tmp_path, backbone, False, seed=1, batch_size=4, clips_per_shard=6)
    history = train(
        tmp_path, tmp_path / "projection_head.pt", hidden_dim=64, out_dim=16, epochs=2, batch_size=4, device="cpu"
    )
    assert len(history) == 2

    table = encode(tmp_path, tmp_path / "projection_head.pt", device="cpu")
    assert table.loc["overall", "n"] == 6 and 0.0 <= table.loc["overall", "top1"] <= 1.0
    references = np.load(tmp_path / "fingerprints" / "references.npz")
    assert references["vectors"].shape == (24, 16)
    np.testing.assert_allclose(np.linalg.norm(references["vectors"], axis=1), 1.0, atol=1e-5)
