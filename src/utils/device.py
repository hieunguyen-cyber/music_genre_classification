from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    is_cuda: bool
    is_mps: bool


def get_device(choice: DeviceChoice = "auto") -> DeviceInfo:
    """
    Auto-detect device with priority: CUDA > MPS > CPU.
    Supports override via choice.
    """
    if choice == "cpu":
        d = torch.device("cpu")
    elif choice == "cuda":
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif choice == "mps":
        d = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    else:
        d = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )

    name = str(d)
    return DeviceInfo(device=d, name=name, is_cuda=d.type == "cuda", is_mps=d.type == "mps")
