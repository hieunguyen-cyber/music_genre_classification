from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for Python, NumPy, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed PyTorch DataLoader workers."""
    # Cast to Python int first to avoid numpy uint32 overflow before the modulo
    worker_seed_int = (int(np.random.get_state()[1][0]) + worker_id) % 2**32
    np.random.seed(worker_seed_int)
    random.seed(worker_seed_int)
