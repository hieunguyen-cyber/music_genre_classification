from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class SklearnTrainResult:
    checkpoint_path: Path
    val_accuracy: float
    val_log_loss: float | None


def fit_and_save(
    model: Any,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    run_dir: Path,
    model_name: str,
) -> SklearnTrainResult:
    """
    Fit an sklearn model and save it as a checkpoint.

    Saves:
      - `checkpoints/model.joblib`
      - `reports/sklearn_train_metrics.json`
    """
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    val_acc = float((val_pred == y_val).mean())

    val_ll = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_val)
            val_ll = float(log_loss(y_val, proba, labels=np.unique(y_train)))
        except Exception:
            val_ll = None

    ckpt = run_dir / "checkpoints" / "model.joblib"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
        },
        ckpt,
    )

    metrics = {"model_name": model_name, "val_accuracy": val_acc, "val_log_loss": val_ll}
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "sklearn_train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return SklearnTrainResult(checkpoint_path=ckpt, val_accuracy=val_acc, val_log_loss=val_ll)


def load_sklearn_checkpoint(path: Path) -> Tuple[str, Any]:
    obj = joblib.load(path)
    return str(obj["model_name"]), obj["model"]
