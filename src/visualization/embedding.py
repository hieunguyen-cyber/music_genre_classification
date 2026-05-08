from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _scatter(z: np.ndarray, y: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        m = y == i
        if np.any(m):
            plt.scatter(z[m, 0], z[m, 1], s=8, alpha=0.7, label=cls)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pca(x: np.ndarray, y: np.ndarray, class_names: list[str], out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(x)
    _scatter(z, y, class_names, out_path, "PCA (2D)")


def plot_tsne(x: np.ndarray, y: np.ndarray, class_names: list[str], out_path: Path, perplexity: float = 30.0) -> None:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
    z = tsne.fit_transform(x)
    _scatter(z, y, class_names, out_path, "t-SNE (2D)")


def plot_umap(x: np.ndarray, y: np.ndarray, class_names: list[str], out_path: Path) -> None:
    try:
        import umap  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("UMAP requires `umap-learn`. Install it or disable in config.") from e

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    z = reducer.fit_transform(x)
    _scatter(z, y, class_names, out_path, "UMAP (2D)")
