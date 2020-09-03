"""Utilities for data manipulation."""

from typing import Union
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import lfilter


def load_wav(audio_path: Union[str, Path], sample_rate: int) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    return wav


def mulaw_encode(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Encode signal based on mu-law companding."""
    assert x.max() < 1.0 and x.min() > -1.0
    mu = n_channels - 1
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    x_mu = np.floor((x_mu + 1) / 2 * mu + 0.5).astype(np.int64)
    return x_mu


def mulaw_decode(x_mu: np.ndarray, n_channels: int) -> np.ndarray:
    """Decode mu-law encoded signal."""
    mu = n_channels - 1
    x = np.sign(x_mu) / mu * ((1 + mu) ** np.abs(x_mu) - 1)
    return x


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T
