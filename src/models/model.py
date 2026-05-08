from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch.nn as nn

from src.models.cnn import CNNMel
from src.models.lstm import LSTMMel


class MLP(nn.Module):
    """MLP classifier for tabular features."""

    def __init__(self, input_dim: int, n_classes: int, hidden_sizes: Sequence[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        layers.append(nn.Dropout(dropout))
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


def create_model(
    model_name: str,
    model_cfg: Dict[str, Any],
    *,
    input_dim: int,
    n_classes: int,
    n_mels: int | None = None,
) -> nn.Module:
    """
    Factory to create a model from config.
    """
    if model_name == "mlp":
        hidden_sizes = model_cfg.get("hidden_sizes", [512, 256, 128, 64, 32])
        dropout = float(model_cfg.get("dropout", 0.2))
        return MLP(input_dim=input_dim, n_classes=n_classes, hidden_sizes=hidden_sizes, dropout=dropout)
    if model_name == "cnn_mel":
        if n_mels is None:
            raise ValueError("n_mels is required for cnn_mel")
        channels = model_cfg.get("channels", [32, 64, 128])
        dropout = float(model_cfg.get("dropout", 0.3))
        return CNNMel(n_mels=n_mels, n_classes=n_classes, channels=channels, dropout=dropout)
    if model_name == "lstm_mel":
        if n_mels is None:
            raise ValueError("n_mels is required for lstm_mel")
        return LSTMMel(
            n_mels=n_mels,
            n_classes=n_classes,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown model name: {model_name}")
