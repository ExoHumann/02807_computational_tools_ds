import io
import zipfile

import numpy as np
import soundfile as sf

from afp.data import SAMPLE_RATE, build_manifest, load_audio, process_zip, segment


def wav_bytes(seconds: float, rate: int = 44_100, channels: int = 2) -> bytes:
    """Return an in-memory wav file with a sine tone."""
    t = np.arange(round(seconds * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone] * channels, axis=1), rate, format="WAV")
    return buffer.getvalue()


def test_segment_30s_clip_gives_five_overlapping_windows():
    waveform = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
    segments, start_s = segment(waveform)
    assert segments.shape == (5, 10 * SAMPLE_RATE)
    np.testing.assert_allclose(start_s, [0.0, 5.0, 10.0, 15.0, 20.0])


def test_load_audio_returns_mono_at_target_rate():
    waveform = load_audio(io.BytesIO(wav_bytes(2.0)))
    assert waveform.dtype == np.float32
    assert waveform.shape == (2 * SAMPLE_RATE,)


def test_process_zip_writes_int16_skips_bad_tracks_and_resumes(tmp_path):
    zip_path = tmp_path / "fake.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("fake/000/000001.mp3", wav_bytes(12.0))
        zf.writestr("fake/000/000002.mp3", wav_bytes(12.0, rate=22_050, channels=1))
        zf.writestr("fake/000/000003.mp3", wav_bytes(3.0))
        zf.writestr("fake/000/000004.mp3", b"not audio")
    processed = tmp_path / "processed"

    skipped = process_zip(zip_path, processed, workers=1)

    assert sorted(skipped["track_id"]) == [3, 4]
    manifest = build_manifest(processed)
    assert manifest["track_id"].tolist() == [1, 2]
    assert (manifest["n_samples"] == 12 * SAMPLE_RATE).all()
    assert np.load(processed / manifest["path"][0]).dtype == np.int16
    assert len(process_zip(zip_path, processed, workers=1)) == 2
