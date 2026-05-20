from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Audio-domain augmentation (waveform level)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature-domain augmentation (tabular / embedding level)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MixupCfg:
    """
    Mixup augmentation for tabular features.

    Mixup interpolates pairs of training samples and their labels:
        x_mix = lambda * x_i + (1 - lambda) * x_j
        y_mix = lambda * y_i + (1 - lambda) * y_j   (soft labels)

    where lambda ~ Beta(alpha, alpha).

    This implicitly regularises the model to output smooth, linear
    interpolations in feature space — reducing overconfidence and providing
    a calibration benefit.  It also acts as a soft form of data augmentation
    that exposes the model to "boundary" examples between genres, directly
    addressing the genre-overlap challenge.

    References
    ----------
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
    """

    enabled: bool = False
    alpha: float = 0.2          # Beta distribution shape; 0.2 is a common default
    prob: float = 1.0           # probability of applying mixup to a batch


def mixup_batch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cfg: MixupCfg,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Mixup to a numpy batch (x, y).

    Parameters
    ----------
    x : float32 array (N, D)
    y : int64 array (N,) — integer class indices

    Returns
    -------
    x_mix : float32 array (N, D)
    y_soft : float32 array (N, C) — soft (one-hot + interpolated) labels
        The caller is responsible for using a soft-label compatible loss
        (e.g. ``F.cross_entropy(logits, y_soft)`` with soft targets, or
        ``(loss_a * lam + loss_b * (1-lam))`` formulation).
    """
    if not cfg.enabled:
        raise RuntimeError("mixup_batch called but MixupCfg.enabled=False")
    rng = rng or np.random.default_rng()
    n = len(x)
    n_classes = int(y.max()) + 1

    lam = float(rng.beta(cfg.alpha, cfg.alpha)) if cfg.alpha > 0 else 1.0
    idx = rng.permutation(n)

    x_mix = lam * x + (1.0 - lam) * x[idx]

    # build soft labels
    y_onehot = np.eye(n_classes, dtype=np.float32)[y]
    y_soft = lam * y_onehot + (1.0 - lam) * y_onehot[idx]

    return x_mix.astype(np.float32), y_soft


@dataclass(frozen=True)
class SpecAugmentCfg:
    """
    SpecAugment for mel spectrogram tensors (time and frequency masking).

    Addresses domain shift by training the model to be robust to
    missing frequency bands and time frames — simulating recording
    environment differences (microphone roll-off, room acoustics, etc.).

    References
    ----------
    Park et al., "SpecAugment: A Simple Data Augmentation Method for
    Automatic Speech Recognition", Interspeech 2019.
    """

    enabled: bool = False
    freq_mask_param: int = 20   # max number of mel bins to mask
    time_mask_param: int = 20   # max number of time frames to mask
    n_freq_masks: int = 2       # number of frequency masks to apply
    n_time_masks: int = 2       # number of time masks to apply


def apply_spec_augment(
    mel: np.ndarray,
    cfg: SpecAugmentCfg,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply SpecAugment masks to a mel spectrogram.

    Parameters
    ----------
    mel : float32 array shape (n_mels, time) or (1, n_mels, time)

    Returns
    -------
    Augmented mel of same shape.
    """
    if not cfg.enabled:
        return mel

    rng = rng or np.random.default_rng()
    squeeze = mel.ndim == 2
    if squeeze:
        mel = mel[np.newaxis]  # (1, n_mels, time)

    out = mel.copy()
    _, n_mels, n_time = out.shape

    # Frequency masking
    for _ in range(cfg.n_freq_masks):
        f = int(rng.integers(0, cfg.freq_mask_param + 1))
        f0 = int(rng.integers(0, max(1, n_mels - f)))
        out[:, f0:f0 + f, :] = 0.0

    # Time masking
    for _ in range(cfg.n_time_masks):
        t = int(rng.integers(0, cfg.time_mask_param + 1))
        t0 = int(rng.integers(0, max(1, n_time - t)))
        out[:, :, t0:t0 + t] = 0.0

    return out.squeeze(0) if squeeze else out
