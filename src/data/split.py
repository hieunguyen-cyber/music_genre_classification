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


def save_split(split_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
