from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import MixupCfg


@dataclass
class TrainHistory:
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    lr: List[float]


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


@dataclass(frozen=True)
class EarlyStoppingCfg:
    enabled: bool
    patience: int
    monitor: str
    mode: str  # min|max


def better(a: float, b: float, mode: str) -> bool:
    return a < b if mode == "min" else a > b


def _mixup_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Mixup loss: lambda * CE(logits, y_a) + (1-lambda) * CE(logits, y_b)."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    run_dir: Path,
    early_stopping: EarlyStoppingCfg,
    checkpoint_metric: str,
    checkpoint_mode: str,
    mixup_cfg: Optional[MixupCfg] = None,
) -> Tuple[nn.Module, TrainHistory, Path]:
    """
    Train a classifier with CrossEntropyLoss, optionally with Mixup.

    Saves:
      - `best.pt` checkpoint
      - `history.json`

    Mixup (Zhang et al. 2018):
      When ``mixup_cfg.enabled=True``, pairs of training samples are linearly
      interpolated (lambda ~ Beta(alpha, alpha)).  The loss is computed as a
      convex combination of the two cross-entropy losses, which regularises
      the model's decision boundaries and reduces overconfidence — directly
      addressing genre-overlap issues.
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    use_mixup = (mixup_cfg is not None) and mixup_cfg.enabled
    rng = np.random.default_rng(42)

    model.to(device)
    best_metric = float("inf") if checkpoint_mode == "min" else float("-inf")
    best_path = run_dir / "checkpoints" / "best.pt"

    history = TrainHistory(epochs=[], train_loss=[], val_loss=[], train_accuracy=[], val_accuracy=[], lr=[])
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_accs: List[float] = []
        pbar = tqdm(train_loader, desc=f"Train {epoch:03d}/{epochs}", unit="batch")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_mixup and rng.random() < (mixup_cfg.prob if mixup_cfg else 1.0):
                # ── Mixup forward pass ─────────────────────────────────────
                alpha = mixup_cfg.alpha  # type: ignore[union-attr]
                lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
                batch_size = x.size(0)
                idx = torch.randperm(batch_size, device=device)
                x_mix = lam * x + (1.0 - lam) * x[idx]
                y_a, y_b = y, y[idx]
                logits = model(x_mix)
                loss = _mixup_loss(logits, y_a, y_b, lam, criterion)
                # Accuracy against the dominant label (for monitoring only)
                acc = accuracy_from_logits(logits.detach(), y_a if lam >= 0.5 else y_b)
            else:
                logits = model(x)
                loss = criterion(logits, y)
                acc = accuracy_from_logits(logits.detach(), y)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_accs.append(acc)
            pbar.set_postfix(
                loss=np.mean(train_losses),
                acc=np.mean(train_accs),
                lr=optimizer.param_groups[0]["lr"],
                mixup=use_mixup,
            )

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val   {epoch:03d}/{epochs}", unit="batch", leave=False):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)  # val always uses standard CE
                val_losses.append(float(loss.item()))
                val_accs.append(accuracy_from_logits(logits, y))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        # scheduler keyed on val_loss
        scheduler.step(val_loss)

        history.epochs.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)
        history.lr.append(float(optimizer.param_groups[0]["lr"]))

        current_metric = val_loss if checkpoint_metric == "val_loss" else val_acc
        if better(current_metric, best_metric, checkpoint_mode):
            best_metric = current_metric
            save_checkpoint(best_path, model, optimizer, epoch=epoch, best_metric=best_metric)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if early_stopping.enabled and bad_epochs >= early_stopping.patience:
            break

    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    hist_path = run_dir / "reports" / "history.json"
    hist_path.write_text(json.dumps(history.__dict__, indent=2), encoding="utf-8")

    # Load best for return
    load_checkpoint(best_path, model)
    return model, history, best_path
