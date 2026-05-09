from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np
from tqdm import tqdm

from src.data.augmentation import AugmentCfg, apply_augmentations


@dataclass(frozen=True)
class MelCfg:
    sample_rate: int
    segment_seconds: float
    n_mels: int
    n_fft: int
    hop_length: int
    fmin: int
    fmax: int


def iter_tracks(raw_genres_dir: Path) -> List[Path]:
    """Return sorted list of GTZAN 30-sec wav paths under genres_original."""
    return sorted(raw_genres_dir.rglob("*.wav"))


def segment_audio(y: np.ndarray, sr: int, segment_seconds: float) -> List[np.ndarray]:
    seg_len = int(round(segment_seconds * sr))
    if seg_len <= 0:
        raise ValueError("segment_seconds too small")
    n_segments = max(1, len(y) // seg_len)
    out = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        if end > len(y):
            break
        out.append(y[start:end])
    return out


def mel_spectrogram(y: np.ndarray, sr: int, cfg: MelCfg) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def extract_mel_dataset(
    raw_genres_dir: Path,
    *,
    cfg: MelCfg,
    augment: AugmentCfg | None = None,
    seed: int = 42,
    limit_tracks: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract mel-spectrogram segments from raw GTZAN audio.

    Returns:
      X: (N, 1, n_mels, time)
      y: (N,)
      groups: (N,) group id (track id) to allow group splitting
      class_names: sorted genre names
    """
    augment = augment or AugmentCfg(enabled=False)
    rng = np.random.default_rng(seed)

    tracks = iter_tracks(raw_genres_dir)
    if limit_tracks is not None:
        tracks = tracks[:limit_tracks]

    genres = sorted({p.parent.name for p in tracks})
    class_names = genres
    genre_to_idx = {g: i for i, g in enumerate(class_names)}

    xs: List[np.ndarray] = []
    ys: List[int] = []
    groups: List[str] = []

    for p in tqdm(tracks, desc="Extract mel", unit="track"):
        genre = p.parent.name
        try:
            y, sr = librosa.load(str(p), sr=cfg.sample_rate, mono=True)
        except Exception as e:
            print(f"Warning: Skipping corrupted file {p}: {e}")
            continue
        segs = segment_audio(y, sr, cfg.segment_seconds)
        group_id = p.stem  # e.g., blues.00000
        for seg in segs:
            seg2 = apply_augmentations(seg, sr, augment, rng=rng)
            m = mel_spectrogram(seg2, sr, cfg)
            xs.append(m[None, :, :])  # (1, n_mels, time)
            ys.append(int(genre_to_idx[genre]))
            groups.append(group_id)

    X = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.array(ys, dtype=np.int64)
    g_arr = np.array(groups, dtype=object)
    return X, y_arr, g_arr, class_names
