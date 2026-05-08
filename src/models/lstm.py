from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class LSTMMel(nn.Module):
    """
    LSTM classifier for mel-spectrogram sequences.

    Expected input: (B, time, n_mels)
    """

    def __init__(
        self,
        n_mels: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
