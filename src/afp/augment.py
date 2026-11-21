"""Audio distortions used to build query/reference pairs, following the paper (Section 2.3).

Differences from the paper, since MUSAN, the Aachen room impulse responses and Encodec are not downloaded:
    * noise is synthetic coloured noise instead of MUSAN recordings,
    * reverb uses a synthetic exponentially decaying impulse response instead of Aachen RIRs,
    * Encodec compression is not implemented.

``TEST_AUGMENTATIONS`` are TikTok-like distortions (sped-up playback, voice-over, MP3 compression) used for held-out
queries. ``VOICE_CATEGORIES`` optionally add voice-over to training, with speech from speakers disjoint from the test
speech, so the test voices are never heard during training.
"""

import io
import math
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from scipy.signal import resample_poly

AUGMENTATIONS = (
    "time_stretch",
    "pitch_shift",
    "noise",
    "reverb",
    "band_pass",
    "high_pass",
    "low_pass",
    "echo",
)

CATEGORIES = (
    ("time_stretch",),
    ("pitch_shift",),
    ("time_stretch", "pitch_shift"),
    ("noise",),
    ("reverb",),
    ("reverb", "noise"),
    ("band_pass",),
    ("high_pass",),
    ("low_pass",),
    ("echo",),
)

TEST_AUGMENTATIONS = ("speed_up", "voice_over", "compression")

VOICE_CATEGORIES = (
    ("voice_over",),
    ("voice_over", "noise"),
)

TEST_CATEGORIES = (
    ("speed_up",),
    ("voice_over",),
    ("compression",),
    ("speed_up", "voice_over", "compression"),
)

PARAMETER_RANGES = {
    "time_stretch": (0.7, 1.5),
    "pitch_shift": (-5.0, 5.0),
    "noise_snr_db": (0.0, 20.0),
    "noise_color": (0.0, 2.0),
    "reverb_rt60_s": (0.2, 1.2),
    "band_pass": (300.0, 1800.0),
    "high_pass": (1800.0, 3400.0),
    "low_pass": (300.0, 1500.0),
    "echo_delay_s": (0.1, 0.2),
    "speed_up": (1.1, 1.5),
    "voice_over_music_to_speech_db": (-5.0, 10.0),
    "compression_level": (0.5, 0.9),
}

ECHO_DECAY = 0.5
STFT_SIZE = 1024
STFT_HOP = 256
PITCH_RATIO_DENOMINATOR = 64


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Return the root-mean-square of ``x`` with a small floor so silence does not divide by zero."""
    return x.pow(2).mean().sqrt().clamp_min(1e-8)


def _time_stretch(x: torch.Tensor, rate: float) -> torch.Tensor:
    """Change speed without changing pitch using a phase vocoder. ``rate > 1`` is faster and shorter."""
    window = torch.hann_window(STFT_SIZE)
    spec = torch.stft(x, STFT_SIZE, STFT_HOP, window=window, return_complex=True)
    phase_advance = torch.linspace(0, math.pi * STFT_HOP, spec.shape[0])[..., None]
    stretched = F.phase_vocoder(spec, rate, phase_advance)
    return torch.istft(stretched, STFT_SIZE, STFT_HOP, window=window)


def _pitch_shift(x: torch.Tensor, semitones: float) -> torch.Tensor:
    """Change pitch without changing duration: time-stretch, then resample back with a small rational ratio.

    ``torchaudio.functional.pitch_shift`` resamples with a ratio that rarely simplifies, which builds an enormous
    filter bank (seconds and gigabytes) for some random shifts. Approximating the ratio by a fraction with a small
    denominator keeps every call fast, at a negligible pitch error.
    """
    rate = 2.0 ** (-semitones / 12.0)
    ratio = Fraction(rate).limit_denominator(PITCH_RATIO_DENOMINATOR)
    shifted = resample_poly(_time_stretch(x, rate).numpy(), ratio.numerator, ratio.denominator)
    out = torch.zeros_like(x)
    n = min(len(x), len(shifted))
    out[:n] = torch.from_numpy(shifted[:n].astype(np.float32))
    return out


def _noise(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Add coloured noise (white to brown) at a random signal-to-noise ratio."""
    low, high = PARAMETER_RANGES["noise_snr_db"]
    snr_db = rng.uniform(low, high)
    low, high = PARAMETER_RANGES["noise_color"]
    color = rng.uniform(low, high)
    spectrum = torch.fft.rfft(torch.from_numpy(rng.standard_normal(len(x)).astype(np.float32)))
    freqs = torch.arange(1, len(spectrum) + 1, dtype=torch.float32)
    noise = torch.fft.irfft(spectrum / freqs.pow(color / 2), n=len(x))
    return x + noise * (_rms(x) / _rms(noise) / 10 ** (snr_db / 20))


def _reverb(x: torch.Tensor, rng: np.random.Generator, sample_rate: int) -> torch.Tensor:
    """Convolve with a synthetic impulse response: a direct path plus noise decaying by 60 dB over the RT60."""
    low, high = PARAMETER_RANGES["reverb_rt60_s"]
    rt60 = rng.uniform(low, high)
    n = int(rt60 * sample_rate)
    decay = torch.exp(-6.91 * torch.arange(n, dtype=torch.float32) / n)
    tail = torch.from_numpy(rng.standard_normal(n).astype(np.float32)) * decay
    rir = 0.3 * tail / tail.norm()
    rir[0] += 1.0
    wet = F.fftconvolve(x, rir)[: len(x)]
    return wet * (_rms(x) / _rms(wet))


def _echo(x: torch.Tensor, rng: np.random.Generator, sample_rate: int) -> torch.Tensor:
    """Add one delayed, attenuated copy of the signal."""
    low, high = PARAMETER_RANGES["echo_delay_s"]
    delay = int(rng.uniform(low, high) * sample_rate)
    delayed = torch.zeros_like(x)
    delayed[delay:] = x[:-delay]
    return x + ECHO_DECAY * delayed


def _speed_up(x: torch.Tensor, rate: float) -> torch.Tensor:
    """Play faster with the pitch rising too, like a "sped up" song: resample by a small rational ratio."""
    ratio = Fraction(1.0 / rate).limit_denominator(PITCH_RATIO_DENOMINATOR)
    return torch.from_numpy(resample_poly(x.numpy(), ratio.numerator, ratio.denominator).astype(np.float32))


def _voice_over(x: torch.Tensor, rng: np.random.Generator, speech: np.ndarray) -> tuple[torch.Tensor, float]:
    """Mix random speech chunks over the whole signal. Returns the mix and the music-to-speech ratio in dB."""
    pieces = [speech[rng.integers(len(speech))] for _ in range(math.ceil(len(x) / speech.shape[1]))]
    voice = torch.from_numpy(np.concatenate(pieces)[: len(x)].astype(np.float32))
    low, high = PARAMETER_RANGES["voice_over_music_to_speech_db"]
    ratio_db = rng.uniform(low, high)
    return x + voice * (_rms(x) / _rms(voice) / 10 ** (ratio_db / 20)), ratio_db


def _compress(x: torch.Tensor, rng: np.random.Generator, sample_rate: int) -> torch.Tensor:
    """Encode to MP3 and decode back. Higher ``compression_level`` means a lower bitrate (about 40 to 27 kbit/s)."""
    buffer = io.BytesIO()
    sf.write(
        buffer,
        x.numpy(),
        sample_rate,
        format="MP3",
        compression_level=rng.uniform(*PARAMETER_RANGES["compression_level"]),
    )
    buffer.seek(0)
    decoded = torch.from_numpy(sf.read(buffer, dtype="float32")[0])
    out = torch.zeros_like(x)
    n = min(len(x), len(decoded))
    out[:n] = decoded[:n]
    return out


def augment(
    waveform: np.ndarray,
    name: str,
    rng: np.random.Generator,
    sample_rate: int,
    speech: np.ndarray | None = None,
    record: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply one named distortion with parameters drawn from ``PARAMETER_RANGES``.

    Ranges are: time stretch factor, pitch shift in semitones, noise SNR in dB and spectral colour, reverb RT60 in
    seconds, filter cut-offs in Hz (band-pass is the pass band, high- and low-pass the cut-off), echo delay in
    seconds, speed-up factor, voice-over music-to-speech ratio in dB and MP3 compression level. Only ``time_stretch``
    and ``speed_up`` change the length (to ``len / rate``).

    Args:
        waveform: 1D ``float32`` waveform in ``[-1, 1]``.
        name: One of ``AUGMENTATIONS`` or ``TEST_AUGMENTATIONS``.
        rng: Random generator used to draw the parameters.
        sample_rate: Sample rate of ``waveform`` in Hz.
        speech: Speech chunks of shape ``[n, samples]`` at ``sample_rate``, needed only for ``voice_over``.
        record: If given, ``voice_over`` stores its music-to-speech ratio here under ``"voice_over_db"``.

    Returns:
        The distorted 1D ``float32`` waveform, scaled down if needed so it does not clip.

    Raises:
        ValueError: If ``name`` is not a known augmentation, or ``voice_over`` is requested without ``speech``.
    """
    x = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    if name == "time_stretch":
        x = _time_stretch(x, rng.uniform(*PARAMETER_RANGES["time_stretch"]))
    elif name == "pitch_shift":
        x = _pitch_shift(x, rng.uniform(*PARAMETER_RANGES["pitch_shift"]))
    elif name == "noise":
        x = _noise(x, rng)
    elif name == "reverb":
        x = _reverb(x, rng, sample_rate)
    elif name == "band_pass":
        low, high = PARAMETER_RANGES["band_pass"]
        x = F.lowpass_biquad(F.highpass_biquad(x, sample_rate, low), sample_rate, high)
    elif name == "high_pass":
        x = F.highpass_biquad(x, sample_rate, rng.uniform(*PARAMETER_RANGES["high_pass"]))
    elif name == "low_pass":
        x = F.lowpass_biquad(x, sample_rate, rng.uniform(*PARAMETER_RANGES["low_pass"]))
    elif name == "echo":
        x = _echo(x, rng, sample_rate)
    elif name == "speed_up":
        x = _speed_up(x, rng.uniform(*PARAMETER_RANGES["speed_up"]))
    elif name == "voice_over":
        if speech is None:
            raise ValueError("voice_over needs speech chunks")
        x, ratio_db = _voice_over(x, rng, speech)
        if record is not None:
            record["voice_over_db"] = ratio_db
    elif name == "compression":
        x = _compress(x, rng, sample_rate)
    else:
        raise ValueError(f"unknown augmentation {name!r}, expected one of {AUGMENTATIONS + TEST_AUGMENTATIONS}")
    return (x / x.abs().max().clamp_min(1.0)).numpy()


def random_augment(
    waveform: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int,
    categories: tuple[tuple[str, ...], ...] = CATEGORIES,
    speech: np.ndarray | None = None,
    record: dict[str, float] | None = None,
) -> tuple[np.ndarray, float, str]:
    """Distort a waveform with a randomly chosen category.

    A category is one distortion or a combination applied in order, mirroring the paper's evaluation groups.

    Args:
        waveform: 1D ``float32`` waveform.
        rng: Random generator.
        sample_rate: Sample rate of ``waveform`` in Hz.
        categories: Categories to choose from: ``CATEGORIES`` (optionally plus ``VOICE_CATEGORIES``) for training and
            the standard queries, ``TEST_CATEGORIES`` for the held-out TikTok-like queries.
        speech: Speech chunks for ``voice_over`` (see :func:`augment`).
        record: Filled with distortion parameters worth keeping (see :func:`augment`).

    Returns:
        A tuple ``(audio, rate, label)``. ``rate`` is ``len(waveform) / len(audio)``, so a time position ``t`` in the
        original is at ``t / rate`` in the distorted audio. ``label`` is the category joined with ``+``.
    """
    names = categories[rng.integers(len(categories))]
    audio = waveform
    for name in names:
        audio = augment(audio, name, rng, sample_rate, speech, record)
    return audio, len(waveform) / len(audio), "+".join(names)
