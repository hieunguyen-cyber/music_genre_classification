from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    split_df: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> SplitResult:
    """
    Create stratified train/val/test splits over rows of df.

    val_size is measured as a fraction of the *remaining* after test split, so that
    final fractions are approximately: test=test_size, val=val_size*(1-test_size).

    WARNING: This function splits by row (segment level). For datasets where multiple
    rows belong to the same recording (e.g. 3-sec segments from the same track), use
    make_group_splits_from_filenames() instead to avoid track-level data leakage.
    """
    y = df[label_column].values
    idx = np.arange(len(df))

    strat = y if stratify else None
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )
    y_trainval = y[trainval_idx]
    strat2 = y_trainval if stratify else None
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=strat2,
    )

    split = np.full(len(df), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    split_df = pd.DataFrame({"index": idx, "split": split})
    return SplitResult(split_df=split_df, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def make_group_splits(
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> dict[str, str]:
    """
    Split by group (e.g., original 30-sec track) to avoid segment leakage.

    Returns:
      mapping group_id -> split ("train"|"val"|"test")
    """
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    # derive one label per group (assumes consistent within group)
    group_labels = np.zeros(len(unique_groups), dtype=labels.dtype)
    for i, g in enumerate(unique_groups):
        group_labels[i] = labels[groups == g][0]

    strat = group_labels if stratify else None
    g_trainval, g_test = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state, stratify=strat
    )
    strat2 = group_labels[np.isin(unique_groups, g_trainval)] if stratify else None
    g_train, g_val = train_test_split(
        g_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=strat2,
    )

    mapping: dict[str, str] = {}
    for g in g_train:
        mapping[str(g)] = "train"
    for g in g_val:
        mapping[str(g)] = "val"
    for g in g_test:
        mapping[str(g)] = "test"
    return mapping


def make_group_splits_from_filenames(
    df: pd.DataFrame,
    *,
    filename_column: str,
    label_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    Group-level split for the tabular CSV track (features_3_sec.csv).

    Filenames are expected in the form ``genre.trackid.segment.wav``
    (e.g. ``blues.00000.0.wav``).  The function extracts the track group as
    ``genre.trackid`` (e.g. ``blues.00000``) and performs a stratified split
    *at track level*, ensuring that all segments of a track fall into exactly
    one split (train / val / test).  This eliminates track-level data leakage
    that occurs when splitting by individual rows/segments.

    Returns
    -------
    pd.DataFrame with columns ``index`` (row position in df) and ``split``
    (one of "train" / "val" / "test").  Drop-in compatible with
    ``build_tabular_cache``.
    """
    # ── extract track group from filename ─────────────────────────────────
    filenames = df[filename_column].astype(str)
    # Pattern: genre.NNNNN.seg.wav  ->  genre.NNNNN
    groups_series = filenames.str.extract(r"^(.+\.\d+)\.\d+\.wav$")[0]

    # Fallback: if extraction fails (non-standard naming), treat each filename as own group
    missing_mask = groups_series.isna()
    if missing_mask.any():
        groups_series[missing_mask] = filenames[missing_mask]

    groups = groups_series.values
    labels = df[label_column].astype(str).values

    # ── group-level split ──────────────────────────────────────────────────
    group_to_split = make_group_splits(
        groups=groups,
        labels=labels,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    # ── map back to individual rows ────────────────────────────────────────
    row_splits = np.array([group_to_split[str(g)] for g in groups], dtype=object)
    idx = np.arange(len(df))
    split_df = pd.DataFrame({"index": idx, "split": row_splits})
    return split_df


def save_split(split_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
