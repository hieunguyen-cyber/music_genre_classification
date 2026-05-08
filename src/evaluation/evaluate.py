from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import EvalResult, compute_metrics


@dataclass(frozen=True)
class Predictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Predictions:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[np.ndarray] = []
    has_proba = True

    for x, y in tqdm(loader, desc="Predict", unit="batch"):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        y_pred.extend([int(v) for v in pred.cpu().numpy().tolist()])
        try:
            y_proba.append(probs.cpu().numpy())
        except Exception:
            has_proba = False

    proba_arr = np.concatenate(y_proba, axis=0) if (has_proba and y_proba) else None
    return Predictions(
        y_true=np.array(y_true, dtype=np.int64),
        y_pred=np.array(y_pred, dtype=np.int64),
        y_proba=proba_arr,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, *, class_names: List[str]) -> Tuple[Predictions, EvalResult]:
    preds = predict(model, loader, device=device)
    res = compute_metrics(preds.y_true, preds.y_pred, class_names=class_names, y_proba=preds.y_proba)
    return preds, res
