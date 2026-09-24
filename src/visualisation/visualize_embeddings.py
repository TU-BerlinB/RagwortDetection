"""
File: src/visualisation/visualize_embeddings.py
Usage:
    python src/visualisation/visualize_embeddings.py
Description:
    Loads precomputed image embeddings and labels, performs dimensionality
    reduction using PCA and UMAP, and saves 2D scatter visualization plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


EMBEDDINGS_PATH = Path("outputs/embeddings.npz")

CLASS_NAMES = {
    0: "Ragwort",
    1: "Other weed",
}


def load_embeddings():
    """Load embeddings and labels from file."""
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    return embeddings, labels


def plot_embeddings(embeddings, labels, title):
    """Create a 2D scatter plot of embeddings."""
    plt.figure(figsize=(10, 8))

    for label, class_name in CLASS_NAMES.items():
        mask = labels == label
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=class_name,
            alpha=0.7,
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("test.png")


def visualize_pca(embeddings, labels):
    """Reduce embeddings to 2D using PCA."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    print(
        "PCA explained variance:",
        pca.explained_variance_ratio_,
    )

    plot_embeddings(
        embeddings_2d,
        labels,
        "DINOv3 Embeddings - PCA",
    )


def visualize_umap(embeddings, labels):
    """Reduce embeddings to 2D using UMAP."""
    umap = UMAP(
        n_components=2,
        random_state=42,
    )
    embeddings_2d = umap.fit_transform(embeddings)

    plot_embeddings(
        embeddings_2d,
        labels,
        "DINOv3 Embeddings - UMAP",
    )


def main():
    embeddings, labels = load_embeddings()

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    visualize_pca(embeddings, labels)
    visualize_umap(embeddings, labels)


if __name__ == "__main__":
    main()
