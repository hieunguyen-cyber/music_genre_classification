from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history_json: Path, out_path: Path) -> None:
    """Plot loss/accuracy curves from `history.json` produced by trainer."""
    hist = json.loads(history_json.read_text(encoding="utf-8"))
    epochs = hist["epochs"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["train_accuracy"], label="train")
    plt.plot(epochs, hist["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
