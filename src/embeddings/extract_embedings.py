
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.dinov3 import DINOv3


# Paths
DATA_DIR = Path("data/dino_dataset")
OUTPUT_PATH = Path("outputs/embeddings.npz")

CLASSES = {
    "ragwort": 0,
    "others": 1,
}


def get_images():
    """Get all images and their labels from the dataset."""

    images = []

    for class_name, label in CLASSES.items():
        class_dir = DATA_DIR / class_name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append((image_path, label))

    return images


def extract_embeddings():
    """Extract one DINOv3 embedding for every image."""

    model = DINOv3()
    images = get_images()

    embeddings = []
    labels = []
    paths = []

    for image_path, label in tqdm(images, desc="Extracting embeddings"):
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            features = model.extract_features(image)

        # Mean pooling over all image tokens
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

