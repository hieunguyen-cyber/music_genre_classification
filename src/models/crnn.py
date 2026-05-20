from __future__ import annotations

"""
CRNN: Convolutional Recurrent Neural Network for mel spectrograms (TODO-5).

Architecture:
  CNN front-end  -> BiLSTM temporal modelling -> Attention pooling -> Linear head

Input shape : (B, 1, n_mels, time)
Output shape: (B, n_classes)

The CNN learns local time-frequency patterns; the BiLSTM captures long-range
temporal dependencies (verse/chorus structure, rhythmic evolution).  Attention
pooling weights each time-step by learned importance, avoiding naive mean
pooling that treats silence and peak moments equally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv2d -> BN -> ReLU -> MaxPool -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int], dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _AttentionPool(nn.Module):
    """Soft attention over time-steps: (B, T, H) -> (B, H)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.attn(x).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=1)         # (B, T)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return out


class CRNNMel(nn.Module):
    """
    CRNN for mel spectrogram classification.

    Parameters
    ----------
    n_mels    : number of mel frequency bins (default 128)
    n_classes : number of output classes
    cnn_channels : output channels for each of the 3 conv blocks
    lstm_hidden  : BiLSTM hidden size per direction
    lstm_layers  : number of stacked LSTM layers
    dropout   : dropout rate (applied in CNN blocks and before classifier)
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 10,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        c1, c2, c3 = cnn_channels

        # CNN front-end: reduce frequency dimension, keep time dimension
        self.cnn = nn.Sequential(
            _ConvBlock(1,  c1, pool=(2, 1), dropout=dropout),  # mel/2, T
            _ConvBlock(c1, c2, pool=(2, 1), dropout=dropout),  # mel/4, T
            _ConvBlock(c2, c3, pool=(2, 1), dropout=dropout),  # mel/8, T
        )

        # After 3 x (mel // 2): freq_out = n_mels // 8
        freq_out = n_mels // 8
        rnn_input_size = c3 * freq_out

        # BiLSTM temporal modelling
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention pooling over time
        self.attn_pool = _AttentionPool(lstm_hidden * 2)  # *2 for bidirectional

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.cnn(x)                        # (B, C3, mel/8, T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)             # (B, T, C, F)
        x = x.reshape(B, T, C * F)            # (B, T, C*F)

        x, _ = self.lstm(x)                    # (B, T, 2*H)
        x = self.attn_pool(x)                  # (B, 2*H)
        return self.head(x)                    # (B, n_classes)
