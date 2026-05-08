from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_duration_distribution(metadata_csv: Path, out_path: Path) -> None:
    """Plot distribution of audio durations from metadata."""
    df = pd.read_csv(metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.hist(df["duration_sec"].values, bins=40)
    plt.title("Audio duration distribution (sec)")
    plt.xlabel("Duration (sec)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sample_rate_distribution(metadata_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    counts = df["sample_rate"].value_counts().sort_index()
    plt.bar([str(x) for x in counts.index], counts.values)
    plt.title("Sample rate distribution")
    plt.xlabel("Sample rate (Hz)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_genre_distribution(metadata_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["genre"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar(counts.index.tolist(), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Genre distribution (raw audio)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
