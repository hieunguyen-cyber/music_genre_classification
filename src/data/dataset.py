from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import SpecAugmentCfg, apply_spec_augment
from src.utils.seed import seed_worker


class NumpyClassificationDataset(Dataset):
    """A simple torch Dataset wrapping (X, y) NumPy arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError(f"X and y length mismatch: {len(x)} != {len(y)}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class MelDataset(Dataset):
    """
    Dataset for mel-spectrogram tensors.

    Inputs:
      - CNN expects: (B, 1, n_mels, time)
      - LSTM expects: (B, time, n_mels) -> set `as_sequence=True`

    SpecAugment (Park et al. 2019):
      When ``spec_augment_cfg.enabled=True`` and ``training=True``, frequency
      and time masks are applied to each mel tensor on-the-fly.  This forces
      the model to rely on the remaining intact signal rather than memorising
      specific frequency/time patterns from a fixed recording environment,
      directly addressing domain shift between GTZAN tracks.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        as_sequence: bool,
        spec_augment_cfg: Optional[SpecAugmentCfg] = None,
        training: bool = False,
    ):
        if len(x) != len(y):
            raise ValueError(f"X and y length mismatch: {len(x)} != {len(y)}")
        self.x = x
        self.y = y
        self.as_sequence = as_sequence
        self.spec_augment_cfg = spec_augment_cfg
        self.training = training
        # Per-instance RNG so each item gets independent augmentation
        self._rng = np.random.default_rng(seed=42)

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np = self.x[idx].copy()  # (1, n_mels, time)

        # Apply SpecAugment only during training
        if (
            self.training
            and self.spec_augment_cfg is not None
            and self.spec_augment_cfg.enabled
        ):
            # x_np shape: (1, n_mels, time) — pass the inner (n_mels, time) slice
            x_np[0] = apply_spec_augment(x_np[0], self.spec_augment_cfg, rng=self._rng)

        x = torch.from_numpy(x_np).float()
        if self.as_sequence:
            # (1, n_mels, T) -> (T, n_mels)
            if x.ndim != 3:
                raise ValueError("Expected mel tensor of shape (1, n_mels, time)")
            x = x.squeeze(0).transpose(0, 1).contiguous()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_dataloaders(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoaders:
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = NumpyClassificationDataset(x_train, y_train)
    val_ds = NumpyClassificationDataset(x_val, y_val)
    test_ds = NumpyClassificationDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


def make_mel_dataloaders(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    as_sequence: bool,
    spec_augment_cfg: Optional[SpecAugmentCfg] = None,
) -> DataLoaders:
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = MelDataset(
        x_train, y_train,
        as_sequence=as_sequence,
        spec_augment_cfg=spec_augment_cfg,
        training=True,       # augment only during training
    )
    val_ds = MelDataset(
        x_val, y_val,
        as_sequence=as_sequence,
        training=False,      # no augmentation for val
    )
    test_ds = MelDataset(
        x_test, y_test,
        as_sequence=as_sequence,
        training=False,      # no augmentation for test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)
