from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.io import save_json


@dataclass(frozen=True)
class TabularCache:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    filenames_train: np.ndarray
    filenames_val: np.ndarray
    filenames_test: np.ndarray
    classes: list[str]


def load_features_csv(path: Path) -> pd.DataFrame:
    """Load GTZAN features CSV (3 sec or 30 sec)."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {path}")
    if "filename" not in df.columns:
        raise ValueError(f"Expected a 'filename' column in {path}")
    return df


def build_tabular_cache(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    *,
    label_column: str,
    filename_column: str,
    drop_columns: list[str],
    standardize: bool,
    out_npz: Path,
    out_scaler: Path,
    out_label_map: Path,
) -> TabularCache:
    """
    Build a cached NumPy dataset from the provided features dataframe and split definition.

    - Fits LabelEncoder and (optionally) StandardScaler on TRAIN split only.
    - Saves: `.npz` arrays, scaler pickle, label map JSON.
    """
    if "index" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError("split_df must contain columns: index, split")

    split_by_index = split_df.set_index("index")["split"].to_dict()
    splits = df.index.map(lambda i: split_by_index.get(int(i), "train"))
    if not set(splits).issuperset({"train", "val", "test"}):
        raise ValueError("Split file must include train/val/test rows")

    filenames = df[filename_column].astype(str).values
    y_str = df[label_column].astype(str).values

    # Prepare X
    x_df = df.drop(columns=drop_columns)
    x = x_df.to_numpy(dtype=np.float32, copy=True)

    # Train-only fit
    train_mask = np.array(splits) == "train"
    val_mask = np.array(splits) == "val"
    test_mask = np.array(splits) == "test"

    le = LabelEncoder()
    le.fit(y_str[train_mask])
    y = le.transform(y_str).astype(np.int64)
    classes = list(le.classes_)

    if standardize:
        scaler = StandardScaler()
        scaler.fit(x[train_mask])
        x = scaler.transform(x).astype(np.float32)
        out_scaler.parent.mkdir(parents=True, exist_ok=True)
        with out_scaler.open("wb") as f:
            pickle.dump(scaler, f)

    label_map = {cls: int(i) for i, cls in enumerate(classes)}
    save_json(label_map, out_label_map)

    cache = TabularCache(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_val=x[val_mask],
        y_val=y[val_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        filenames_train=filenames[train_mask],
        filenames_val=filenames[val_mask],
        filenames_test=filenames[test_mask],
        classes=classes,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x_train=cache.x_train,
        y_train=cache.y_train,
        x_val=cache.x_val,
        y_val=cache.y_val,
        x_test=cache.x_test,
        y_test=cache.y_test,
        filenames_train=cache.filenames_train,
        filenames_val=cache.filenames_val,
        filenames_test=cache.filenames_test,
        classes=np.array(cache.classes),
    )
    return cache


def load_tabular_cache(path: Path) -> TabularCache:
    """Load cached `.npz` produced by build_tabular_cache()."""
    z = np.load(path, allow_pickle=True)
    classes = [str(x) for x in z["classes"].tolist()]
    return TabularCache(
        x_train=z["x_train"],
        y_train=z["y_train"],
        x_val=z["x_val"],
        y_val=z["y_val"],
        x_test=z["x_test"],
        y_test=z["y_test"],
        filenames_train=z["filenames_train"],
        filenames_val=z["filenames_val"],
        filenames_test=z["filenames_test"],
        classes=classes,
    )


@dataclass(frozen=True)
class MelCache:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_val: np.ndarray
    groups_test: np.ndarray
    classes: list[str]


def save_mel_cache(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_by_group: dict[str, str],
    classes: list[str],
    out_npz: Path,
) -> MelCache:
    splits = np.array([split_by_group[str(g)] for g in groups], dtype=object)
    train_m = splits == "train"
    val_m = splits == "val"
    test_m = splits == "test"

    cache = MelCache(
        x_train=x[train_m],
        y_train=y[train_m],
        x_val=x[val_m],
        y_val=y[val_m],
        x_test=x[test_m],
        y_test=y[test_m],
        groups_train=groups[train_m],
        groups_val=groups[val_m],
        groups_test=groups[test_m],
        classes=classes,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x_train=cache.x_train,
        y_train=cache.y_train,
        x_val=cache.x_val,
        y_val=cache.y_val,
        x_test=cache.x_test,
        y_test=cache.y_test,
        groups_train=cache.groups_train,
        groups_val=cache.groups_val,
        groups_test=cache.groups_test,
        classes=np.array(cache.classes),
    )
    return cache


def load_mel_cache(path: Path) -> MelCache:
    z = np.load(path, allow_pickle=True)
    classes = [str(x) for x in z["classes"].tolist()]
    return MelCache(
        x_train=z["x_train"],
        y_train=z["y_train"],
        x_val=z["x_val"],
        y_val=z["y_val"],
        x_test=z["x_test"],
        y_test=z["y_test"],
        groups_train=z["groups_train"],
        groups_val=z["groups_val"],
        groups_test=z["groups_test"],
        classes=classes,
    )
