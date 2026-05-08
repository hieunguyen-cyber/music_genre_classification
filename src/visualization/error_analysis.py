from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfusionPair:
    true_label: int
    pred_label: int
    count: int


def top_confusions(cm: np.ndarray, top_k: int = 10) -> List[ConfusionPair]:
    """Return the largest off-diagonal confusions."""
    pairs: List[ConfusionPair] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            pairs.append(ConfusionPair(true_label=i, pred_label=j, count=int(cm[i, j])))
    pairs.sort(key=lambda p: p.count, reverse=True)
    return pairs[:top_k]


def build_misclassified_table(
    filenames: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_csv: Path,
    max_rows: int = 200,
) -> pd.DataFrame:
    """
    Save a CSV of misclassified samples for inspection.
    """
    m = y_true != y_pred
    df = pd.DataFrame(
        {
            "filename": filenames[m].astype(str),
            "true": [class_names[int(i)] for i in y_true[m]],
            "pred": [class_names[int(i)] for i in y_pred[m]],
        }
    )
    df = df.head(max_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def segment_filename_to_raw(filename: str) -> str:
    """
    Map GTZAN 3-sec filename like `blues.00000.0.wav` -> `blues.00000.wav`.
    """
    parts = filename.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:3])  # genre, id, wav
    return filename


def raw_audio_path(raw_genres_dir: Path, filename: str, genre: str | None = None) -> Path:
    """
    Resolve a raw audio path from filename. If genre is not provided, infer it from filename prefix.
    """
    if genre is None:
        genre = filename.split(".")[0]
    return raw_genres_dir / genre / filename
