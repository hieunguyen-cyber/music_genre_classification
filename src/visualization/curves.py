from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """
    Plot one-vs-rest ROC curves for each class.

    Requires `y_proba` of shape (N, C).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        try:
            RocCurveDisplay.from_predictions(y_bin, y_proba[:, i], name=class_names[i], ax=ax)
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve (OvR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiclass_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """
    Plot one-vs-rest precision-recall curves for each class.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        try:
            PrecisionRecallDisplay.from_predictions(y_bin, y_proba[:, i], name=class_names[i], ax=ax)
        except Exception:
            continue
    ax.set_title("Precision-Recall Curve (OvR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
