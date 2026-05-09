from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AugmentCfg:
    """
    Audio augmentation configuration.

    This module is optional for the current tabular-CSV pipeline, but enables
    easy extension when extracting features directly from raw waveforms.
    """

    enabled: bool = False
    noise_std: float = 0.005
    time_stretch_low: float = 0.9
    time_stretch_high: float = 1.1
    pitch_shift_steps_low: int = -2
    pitch_shift_steps_high: int = 2


def add_noise(y: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise."""
    return y + rng.normal(0.0, std, size=y.shape).astype(y.dtype)


def time_stretch(y: np.ndarray, rate: float, *, sr: int) -> np.ndarray:
    """Time-stretch audio (requires librosa)."""
    import librosa

    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, *, sr: int, n_steps: int) -> np.ndarray:
    """Pitch-shift audio by semitones (requires librosa)."""
    import librosa

    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def apply_augmentations(y: np.ndarray, sr: int, cfg: AugmentCfg, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply a random composition of augmentations.
    """
    if not cfg.enabled:
        return y
    rng = rng or np.random.default_rng()

    out = y
    # Noise
    if cfg.noise_std > 0 and rng.random() < 0.5:
        out = add_noise(out, cfg.noise_std, rng)
    # Time stretch
    if rng.random() < 0.5:
        rate = float(rng.uniform(cfg.time_stretch_low, cfg.time_stretch_high))
        out = time_stretch(out, rate=rate, sr=sr)
    # Pitch shift
    if rng.random() < 0.5:
        steps = int(rng.integers(cfg.pitch_shift_steps_low, cfg.pitch_shift_steps_high + 1))
        if steps != 0:
            out = pitch_shift(out, sr=sr, n_steps=steps)
    return out
