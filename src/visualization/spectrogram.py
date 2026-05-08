from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_stft(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = None, n_fft: int = 2048, hop_length: int = 512) -> None:
    """Plot STFT magnitude spectrogram (dB)."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"STFT (dB): {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mel_spectrogram(
    audio_path: Path,
    out_path: Path,
    *,
    sample_rate: Optional[int] = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 8000,
) -> None:
    """Plot mel spectrogram (dB)."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram (dB): {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mfcc(
    audio_path: Path,
    out_path: Path,
    *,
    sample_rate: Optional[int] = 22050,
    n_mfcc: int = 20,
    n_mels: int = 128,
    hop_length: int = 512,
) -> None:
    """Plot MFCC heatmap."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title(f"MFCC: {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_chromagram(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = 22050, hop_length: int = 512) -> None:
    """Plot chromagram."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 3.5))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title(f"Chromagram: {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spectral_contrast(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = 22050, hop_length: int = 512) -> None:
    """Plot spectral contrast."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 3.5))
    librosa.display.specshow(contrast, x_axis="time", sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title(f"Spectral Contrast: {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tempogram(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = 22050, hop_length: int = 512) -> None:
    """Plot tempogram (rhythm)."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(tempo, x_axis="time", y_axis="tempo", sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title(f"Tempogram: {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hpss(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = 22050, n_fft: int = 2048, hop_length: int = 512) -> None:
    """Visualize harmonic/percussive separation as two spectrograms."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    H, P = librosa.decompose.hpss(S)
    H_db = librosa.amplitude_to_db(np.abs(H), ref=np.max)
    P_db = librosa.amplitude_to_db(np.abs(P), ref=np.max)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(H_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Harmonic (dB)")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(P_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Percussive (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
