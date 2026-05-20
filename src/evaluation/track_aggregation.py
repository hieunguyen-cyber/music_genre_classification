from __future__ import annotations

"""
Track-level prediction aggregation (TODO-3).

Segment-level models produce one prediction per 3-second clip, but real-world
genre recognition is a *song-level* task.  This module aggregates segment
predictions belonging to the same parent track into a single track-level
prediction using two strategies:

  majority_vote  — most frequent segment label wins.
  mean_proba     — average per-class probabilities, then argmax.

Usage (from stage_evaluate in main.py):

    from src.evaluation.track_aggregation import aggregate_by_track, track_eval_report

    track_preds = aggregate_by_track(
        groups=cache.groups_test,
        y_true=preds.y_true,
        y_pred=preds.y_pred,
        y_proba=preds.y_proba,
        mode="mean_proba",
    )
    report_text = track_eval_report(track_preds, class_names=cache.classes)
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class TrackPredictions:
    """Aggregated track-level predictions."""
    track_ids: np.ndarray        # (T,) unique track group IDs
    y_true: np.ndarray           # (T,) true label per track
    y_pred: np.ndarray           # (T,) predicted label per track
    y_proba: Optional[np.ndarray]  # (T, C) averaged probabilities, or None
    mode: str                    # "majority_vote" | "mean_proba"


def aggregate_by_track(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    *,
    mode: Literal["majority_vote", "mean_proba"] = "mean_proba",
) -> TrackPredictions:
    """
    Aggregate segment-level predictions to track level.

    Parameters
    ----------
    groups : (N,) array of track group IDs (e.g. "blues.00000")
    y_true : (N,) integer class labels (segment level)
    y_pred : (N,) predicted class labels (segment level)
    y_proba : (N, C) predicted probabilities, or None
    mode : aggregation strategy

    Returns
    -------
    TrackPredictions with one entry per unique track.
    """
    unique_tracks = np.unique(groups)
    track_y_true: List[int] = []
    track_y_pred: List[int] = []
    track_y_proba: List[np.ndarray] = []

    for tid in unique_tracks:
        mask = groups == tid
        # True label: take the majority within this track (should be all the same)
        labels_true = y_true[mask]
        true_label = int(np.bincount(labels_true).argmax())
        track_y_true.append(true_label)

        if mode == "majority_vote":
            preds_seg = y_pred[mask]
            track_pred = int(np.bincount(preds_seg).argmax())
            track_y_pred.append(track_pred)
            if y_proba is not None:
                track_y_proba.append(y_proba[mask].mean(axis=0))

        elif mode == "mean_proba":
            if y_proba is not None:
                mean_p = y_proba[mask].mean(axis=0)
                track_y_proba.append(mean_p)
                track_y_pred.append(int(mean_p.argmax()))
            else:
                # Fallback to majority vote when no proba available
                preds_seg = y_pred[mask]
                track_y_pred.append(int(np.bincount(preds_seg).argmax()))
        else:
            raise ValueError(f"Unknown aggregation mode: {mode!r}")

    proba_arr = np.stack(track_y_proba, axis=0) if track_y_proba else None
    return TrackPredictions(
        track_ids=unique_tracks,
        y_true=np.array(track_y_true, dtype=np.int64),
        y_pred=np.array(track_y_pred, dtype=np.int64),
        y_proba=proba_arr,
        mode=mode,
    )


def track_eval_report(tp: TrackPredictions, *, class_names: List[str]) -> str:
    """Build a markdown evaluation report for track-level predictions."""
    acc = float(accuracy_score(tp.y_true, tp.y_pred))
    n_tracks = len(tp.y_true)
    rep = classification_report(
        tp.y_true, tp.y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(tp.y_true, tp.y_pred)
    cm_str = "\n".join(
        "  " + " ".join(f"{v:4d}" for v in row) for row in cm
    )
    roc_str = ""
    if tp.y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            roc = float(roc_auc_score(tp.y_true, tp.y_proba, multi_class="ovr", average="macro"))
            roc_str = f"- ROC-AUC (OvR, macro): **{roc:.4f}**\n"
        except Exception:
            pass

    return (
        f"# Track-Level Evaluation Report\n\n"
        f"**Aggregation mode:** `{tp.mode}`  \n"
        f"**Total tracks:** {n_tracks}\n\n"
        f"## Summary\n\n"
        f"- Track-level Accuracy: **{acc:.4f}**\n"
        + roc_str
        + f"\n## Classification Report\n\n```\n{rep}\n```\n\n"
        f"## Confusion Matrix\n\n"
        f"Rows = true, Cols = predicted (classes: {', '.join(class_names)})\n\n"
        f"```\n{cm_str}\n```\n"
    )
