from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(audio_path: Path, out_path: Path, *, sample_rate: Optional[int] = None, title: str | None = None) -> None:
    """Plot waveform amplitude over time."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    t = np.arange(len(y)) / float(sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 3))
    plt.plot(t, y, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title or f"Waveform: {audio_path.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
