from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_distribution(labels: np.ndarray, class_names: list[str], out_path: Path, title: str = "Class distribution") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar([class_names[i] for i in counts.index], counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_correlation(x: np.ndarray, feature_names: list[str], out_path: Path, max_features: int = 40) -> None:
    """
    Correlation heatmap over a subset of features (to keep plot readable).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_features, x.shape[1])
    df = pd.DataFrame(x[:, :n], columns=feature_names[:n])
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=False)
    plt.title(f"Feature correlation (first {n} features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
