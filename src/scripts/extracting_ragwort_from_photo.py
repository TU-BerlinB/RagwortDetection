"""
File: src/scripts/extracting_ragwort_from_photo.py
Usage:
    python src/scripts/extracting_ragwort_from_photo.py
Description:
    Generates synthetic training images for ragwort detection by cropping real
    ragwort instances from annotated photos and blending them onto background
    landscape images with distance-based edge feathering and updated YOLO labels.
"""

from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm

RAGWORT_IMAGES = Path("data/combined_dataset/train/images")
RAGWORT_LABELS = Path("data/combined_dataset/train/labels")
BACKGROUND_DIR = Path("data/backgrounds/accepted")
OUTPUT_DIR = Path("data/synthetic_dataset")

OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"

NUM_SYNTHETIC_IMAGES = 12000
MIN_SCALE = 0.15
MAX_SCALE = 0.30
MIN_RAGWORTS = 1
MAX_RAGWORTS = 2

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_images(directory: Path):
    return [
        p for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_yolo_labels(label_path: Path):
    """Read YOLO labels and return list of (class_id, xc, yc, w, h)."""
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 5:
                continue
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])
            labels.append((class_id, x_center, y_center, width, height))

    return labels


def get_random_ragwort_crop(ragwort_images: list[Path]):
    """Select a random ragwort image and crop the region around a labeled ragwort plant."""
    while True:
        image_path = random.choice(ragwort_images)
        label_path = RAGWORT_LABELS / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        label = random.choice(labels)
        class_id, xc, yc, bw, bh = label

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        h, w = image.shape[:2]
        x_center = xc * w
        y_center = yc * h
        box_width = bw * w
        box_height = bh * h

        x1 = max(0, int(x_center - box_width / 2))
        y1 = max(0, int(y_center - box_height / 2))
        x2 = min(w, int(x_center + box_width / 2))
        y2 = min(h, int(y_center + box_height / 2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        return crop


def resize_ragwort(crop: np.ndarray, background_width: int):
    """Resize ragwort crop to a random scale relative to background width."""
    scale = random.uniform(MIN_SCALE, MAX_SCALE)
    target_width = int(background_width * scale)
    crop_h, crop_w = crop.shape[:2]
    ratio = target_width / crop_w
    target_height = int(crop_h * ratio)

    return cv2.resize(
        crop,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def paste_ragwort(background: np.ndarray, ragwort: np.ndarray):
    """
    Paste ragwort at a random position with smooth distance-based edge feathering.
    Returns modified background and bounding box (x, y, w, h).
    """
    bg_h, bg_w = background.shape[:2]
    rh, rw = ragwort.shape[:2]

    if rw >= bg_w or rh >= bg_h:
        scale = min((bg_w - 2) / rw, (bg_h - 2) / rh)
        rw = max(1, int(rw * scale))
        rh = max(1, int(rh * scale))
        ragwort = cv2.resize(ragwort, (rw, rh), interpolation=cv2.INTER_AREA)

    max_x = bg_w - rw
    x = random.randint(0, max_x)
    max_y = bg_h - rh
    y_ratio = random.random() ** 0.35
    y = int(y_ratio * max_y)

    # Compute distance-to-edge feathering mask
    distance_x = np.minimum(np.arange(rw), np.arange(rw)[::-1])
    distance_y = np.minimum(np.arange(rh), np.arange(rh)[::-1])
    distance = np.minimum.outer(distance_y, distance_x).astype(np.float32)

    feather_width = random.randint(30, 70)
    mask = np.clip(distance / feather_width, 0.0, 1.0)
    # Smoothstep interpolation
    mask = mask * mask * (3.0 - 2.0 * mask)
    mask = mask[..., np.newaxis]

    background_region = background[y:y + rh, x:x + rw].astype(np.float32)
    ragwort_float = ragwort.astype(np.float32)

    blended = ragwort_float * mask + background_region * (1.0 - mask)
    background[y:y + rh, x:x + rw] = np.clip(blended, 0, 255).astype(np.uint8)

    return background, (x, y, rw, rh)


def bbox_to_yolo(x, y, width, height, image_width, image_height):
    """Convert pixel bounding box to normalized YOLO format."""
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    return (x_center, y_center, width_norm, height_norm)


def main():
    print("=" * 60)
    print("SYNTHETIC RAGWORT DATASET GENERATOR")
    print("=" * 60)

    ragwort_images = get_images(RAGWORT_IMAGES)
    background_images = get_images(BACKGROUND_DIR)

    print(f"Ragwort images:    {len(ragwort_images)}")
    print(f"Background images: {len(background_images)}")
    print(f"Synthetic images:  {NUM_SYNTHETIC_IMAGES}")

    if not ragwort_images:
        raise RuntimeError("No ragwort images found.")
    if not background_images:
        raise RuntimeError("No background images found.")

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    for index in tqdm(range(NUM_SYNTHETIC_IMAGES), desc="Generating"):
        background_path = random.choice(background_images)
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            continue

        bg_h, bg_w = background.shape[:2]
        synthetic = background.copy()
        annotations = []
        num_ragworts = random.randint(MIN_RAGWORTS, MAX_RAGWORTS)

        for _ in range(num_ragworts):
            crop = get_random_ragwort_crop(ragwort_images)
            crop = resize_ragwort(crop, bg_w)
            synthetic, bbox = paste_ragwort(synthetic, crop)

            x, y, width, height = bbox
            xc, yc, bw, bh = bbox_to_yolo(x, y, width, height, bg_w, bg_h)
            annotations.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        filename = f"synthetic_{index:06d}.jpg"
        image_path = OUTPUT_IMAGES / filename
        label_path = OUTPUT_LABELS / filename.replace(".jpg", ".txt")

        cv2.imwrite(str(image_path), synthetic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(annotations))

    print("\n" + "=" * 60)
    print("SYNTHETIC DATASET GENERATION COMPLETED")
    print("=" * 60)
    print(f"Images: {OUTPUT_IMAGES}")
    print(f"Labels: {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()