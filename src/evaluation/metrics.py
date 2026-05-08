from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: str
    confusion: np.ndarray
    roc_auc_ovr_macro: float | None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, class_names: List[str], y_proba: np.ndarray | None = None) -> EvalResult:
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    return EvalResult(
        accuracy=acc,
        macro_f1=float(f1),
        weighted_f1=float(f1w),
        report=rep,
        confusion=cm,
        roc_auc_ovr_macro=roc_auc,
    )
