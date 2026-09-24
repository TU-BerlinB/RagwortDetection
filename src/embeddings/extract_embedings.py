"""
File: src/embeddings/extract_embedings.py
Usage:
    python -m src.embeddings.extract_embedings
    # or: python src/embeddings/extract_embedings.py
Description:
    Extracts feature embeddings for all dataset images using DINOv3
    and saves the embeddings, labels, and image paths to an NPZ archive.
"""

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.models.dinov3 import DINOv3
except ImportError:
    from models.dinov3 import DINOv3

DATA_DIR = Path("data/dino_dataset")
OUTPUT_PATH = Path("outputs/embeddings.npz")

CLASSES = {
    "ragwort": 0,
    "others": 1,
}


def get_images():
    """Get all images and their labels from the dataset directory."""
    images = []

    for class_name, label in CLASSES.items():
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            continue

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append((image_path, label))

    return images


def extract_embeddings():
    """Extract DINOv3 embeddings for every image using mean pooling over tokens."""
    model = DINOv3()
    images = get_images()

    if not images:
        print(f"No images found in {DATA_DIR}")
        return

    embeddings = []
    labels = []
    paths = []

    for image_path, label in tqdm(images, desc="Extracting embeddings"):
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            features = model.extract_features(image)

        embedding = features.mean(dim=1).squeeze(0)

        embeddings.append(embedding.cpu().numpy())
        labels.append(label)
        paths.append(str(image_path))

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_PATH,
        embeddings=embeddings,
        labels=labels,
        paths=np.array(paths),
    )

    print(f"Saved embeddings to: {OUTPUT_PATH}")
    print(f"Number of images: {len(embeddings)}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    extract_embeddings()
