from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import soundfile as sf
from tqdm import tqdm


@dataclass(frozen=True)
class AudioMeta:
    path: Path
    genre: str
    filename: str
    sample_rate: int
    frames: int
    duration_sec: float


def iter_audio_files(raw_root: Path) -> Iterable[Path]:
    """Yield `.wav` files under a GTZAN `genres_original` directory (any depth)."""
    for p in raw_root.rglob("*.wav"):
        yield p


def build_metadata(raw_genres_dir: Path, out_csv: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Build metadata CSV for raw audio.

    Expected layout:
      raw_genres_dir/<genre>/<file>.wav
    """
    rows = []
    files = list(iter_audio_files(raw_genres_dir))
    if limit is not None:
        files = files[:limit]

    for p in tqdm(files, desc="Scanning audio", unit="file"):
        genre = p.parent.name
        filename = p.name
        try:
            info = sf.info(str(p))
            sr = int(info.samplerate)
            frames = int(info.frames)
            dur = float(info.duration)
            ok = True
            err = ""
        except Exception as e1:
            # Some GTZAN mirrors contain a few files that libsndfile can't parse.
            # Try librosa (audioread) as a secondary path; otherwise mark as corrupted.
            try:
                import librosa

                y, sr = librosa.load(str(p), sr=None, mono=True)
                frames = int(len(y))
                dur = frames / float(sr)
                ok = True
                err = f"soundfile_failed: {type(e1).__name__}"
            except Exception as e2:
                sr = 0
                frames = 0
                dur = 0.0
                ok = False
                err = f"unreadable: {type(e1).__name__} / {type(e2).__name__}"
        rows.append(
            dict(
                path=str(p),
                genre=genre,
                filename=filename,
                sample_rate=sr,
                frames=frames,
                duration_sec=dur,
                ok=ok,
                error=err,
            )
        )

    df = pd.DataFrame(rows).sort_values(["genre", "filename"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
