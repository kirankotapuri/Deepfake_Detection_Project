"""
t-SNE Feature Visualization for Deepfake Detection

Visualizes feature embeddings: real vs fake clusters.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def tsne_visualize(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply t-SNE to feature embeddings.

    Args:
        features: Feature matrix [N, D].
        labels: Labels [N] (0=real, 1=fake).
        n_components: Output dimensions (2 for plot).
        perplexity: t-SNE perplexity.
        n_iter: Number of iterations.
        random_state: Random seed.

    Returns:
        Embedded coordinates [N, n_components].
    """
    if not TSNE_AVAILABLE:
        raise ImportError("scikit-learn is required for t-SNE")

    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(features) - 1), max_iter=n_iter, random_state=random_state)
    return tsne.fit_transform(features)


def plot_tsne(
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE: Real vs Fake",
    class_names: tuple[str, ...] = ("Real", "Fake"),
) -> None:
    """
    Plot t-SNE embedding colored by label.

    Args:
        embedding: 2D coordinates [N, 2].
        labels: Binary labels [N].
        save_path: Path to save figure.
        title: Plot title.
        class_names: Labels for legend.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=name, alpha=0.6, s=20)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
