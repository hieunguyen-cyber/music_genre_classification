from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """Resolve project root as the current working directory containing `configs/`."""
    cwd = Path.cwd().resolve()
    if (cwd / "configs").exists():
        return cwd
    # Fallback: climb up a bit
    for p in [cwd] + list(cwd.parents)[:5]:
        if (p / "configs").exists():
            return p
    return cwd


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    reports_dir: Path


def make_run_paths(outputs_root: Path, run_name: str) -> RunPaths:
    run_dir = outputs_root / run_name
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"
    for d in [logs_dir, checkpoints_dir, figures_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
    )
