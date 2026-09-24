"""
File: src/scripts/split_folder.py
Usage:
    python src/scripts/split_folder.py
Description:
    Splits synthetic dataset images and corresponding YOLO annotation files into balanced subdirectories (parts).
"""

from pathlib import Path
import random
import shutil

INPUT_DIR = Path("data/synthetic_dataset")
OUTPUT_DIR = Path("data/synthetic_dataset_split")
NUM_PARTS = 4
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

random.seed(RANDOM_SEED)

INPUT_IMAGES = INPUT_DIR / "images"
INPUT_LABELS = INPUT_DIR / "labels"

images = [
    path
    for path in INPUT_IMAGES.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]

if not images:
    raise RuntimeError(f"No images found in {INPUT_IMAGES}")

random.shuffle(images)

valid_pairs = []
for image_path in images:
    label_path = INPUT_LABELS / f"{image_path.stem}.txt"
    if not label_path.exists():
        print(f"WARNING: Missing label for {image_path.name}")
        continue
    valid_pairs.append((image_path, label_path))

print(f"Total images: {len(images)}")
print(f"Valid image/label pairs: {len(valid_pairs)}")
print(f"Number of parts: {NUM_PARTS}\n")

for part in range(1, NUM_PARTS + 1):
    (OUTPUT_DIR / f"part_{part}" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"part_{part}" / "labels").mkdir(parents=True, exist_ok=True)

total = len(valid_pairs)
base_size = total // NUM_PARTS
remainder = total % NUM_PARTS

start = 0
for part in range(1, NUM_PARTS + 1):
    part_size = base_size + (1 if part <= remainder else 0)
    end = start + part_size
    part_pairs = valid_pairs[start:end]

    print(f"Part {part}: {len(part_pairs)} images")

    output_images = OUTPUT_DIR / f"part_{part}" / "images"
    output_labels = OUTPUT_DIR / f"part_{part}" / "labels"

    for image_path, label_path in part_pairs:
        shutil.copy2(image_path, output_images / image_path.name)
        shutil.copy2(label_path, output_labels / label_path.name)

    start = end

print(f"\nDone. Output written to: {OUTPUT_DIR}")