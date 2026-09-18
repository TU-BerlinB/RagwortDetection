#!/usr/bin/env python3
"""Convert a YOLO detection dataset to COCO Detection JSON.

The source dataset is never modified.  If it has no ``val``/``valid`` split,
``--val-from-train`` reserves a deterministic subset of train *in the output*.

Example (from RagwortDetection):
    python3 src/scripts/yolo_to_coco.py \
        --source data/combined_dataset \
        --output outputs/combined_dataset_coco \
        --exclude-class objects --val-from-train 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "combined_dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "combined_dataset_coco")
    parser.add_argument("--val-from-train", type=float, default=0.2,
                        help="Fraction reserved from train only when no val/valid split exists (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the output-only train/val split.")
    parser.add_argument("--exclude-class", action="append", default=[],
                        help="Class name to omit; repeat the option for multiple classes.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()
    if not 0.0 < args.val_from_train < 1.0:
        parser.error("--val-from-train must be in (0, 1)")
    return args


def load_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda key: int(key))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{data_yaml} must contain a non-empty 'names' list or mapping")
    return [str(name) for name in names]


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def discover_splits(source: Path) -> dict[str, list[Path]]:
    splits = {name: images_in(source / name / "images") for name in ("train", "val", "valid", "test")}
    if not splits["train"]:
        raise FileNotFoundError(f"No supported images found in {source / 'train' / 'images'}")
    if splits["val"] and splits["valid"]:
        raise ValueError("Both val and valid splits exist; rename one to avoid ambiguous output.")
    return splits


def read_yolo_labels(label_path: Path, image_width: int, image_height: int, valid_ids: set[int], counters: Counter) -> list[dict]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        counters["empty_or_missing_labels"] += 1
        return []
    annotations: list[dict] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        # The combined source contains both YOLO detection rows (class, xc,
        # yc, w, h) and YOLO segmentation rows (class followed by polygon
        # x/y pairs).  COCO Detection needs a bounding box, so polygon rows
        # are safely reduced to their enclosing box.
        is_detection = len(values) == 5
        is_segmentation = len(values) >= 7 and (len(values) - 1) % 2 == 0
        if not (is_detection or is_segmentation):
            counters["malformed_labels"] += 1
            print(f"[warning] Skipping malformed label {label_path}:{line_number}")
            continue
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError:
            counters["malformed_labels"] += 1
            print(f"[warning] Skipping non-numeric label {label_path}:{line_number}")
            continue
        if class_id not in valid_ids:
            counters["excluded_or_unknown_labels"] += 1
            continue
        if is_detection:
            x_center, y_center, width, height = coordinates
            x_min = (x_center - width / 2.0) * image_width
            y_min = (y_center - height / 2.0) * image_height
            box_width, box_height = width * image_width, height * image_height
        else:
            x_values, y_values = coordinates[::2], coordinates[1::2]
            x_min, y_min = min(x_values) * image_width, min(y_values) * image_height
            box_width = (max(x_values) - min(x_values)) * image_width
            box_height = (max(y_values) - min(y_values)) * image_height
            counters["segmentation_rows_as_boxes"] += 1
        x0, y0 = max(0.0, x_min), max(0.0, y_min)
        x1, y1 = min(float(image_width), x_min + box_width), min(float(image_height), y_min + box_height)
        if x1 <= x0 or y1 <= y0:
            counters["invalid_boxes"] += 1
            print(f"[warning] Skipping empty/outside box {label_path}:{line_number}")
            continue
        annotations.append({"category_id": class_id + 1, "bbox": [x0, y0, x1 - x0, y1 - y0]})
    return annotations


def source_label(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def convert_split(name: str, images: Iterable[Path], output: Path, categories: list[dict], valid_ids: set[int]) -> tuple[int, int, Counter]:
    split_dir = output / name
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": categories}
    counters: Counter = Counter()
    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        destination = image_dir / image_path.name
        if destination.exists():
            raise FileExistsError(f"Duplicate output image name: {destination}")
        shutil.copy2(image_path, destination)
        with Image.open(image_path) as image:
            width, height = image.size
        coco["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for annotation in read_yolo_labels(source_label(image_path), width, height, valid_ids, counters):
            bbox = annotation["bbox"]
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": annotation["category_id"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            annotation_id += 1
    with (split_dir / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)
    return len(coco["images"]), len(coco["annotations"]), counters


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing {source / 'data.yaml'}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
        shutil.rmtree(output)

    class_names = load_class_names(source / "data.yaml")
    excluded = {name.casefold() for name in args.exclude_class}
    kept = [(index, name) for index, name in enumerate(class_names) if name.casefold() not in excluded]
    if not kept:
        raise ValueError("All classes were excluded")
    categories = [{"id": index + 1, "name": name, "supercategory": "none"} for index, name in kept]
    valid_ids = {index for index, _ in kept}

    splits = discover_splits(source)
    val_images = splits["val"] or splits["valid"]
    train_images = splits["train"]
    if not val_images:
        shuffled = train_images.copy()
        random.Random(args.seed).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * args.val_from_train))
        val_set = set(shuffled[:val_count])
        val_images = sorted(val_set)
        train_images = [path for path in train_images if path not in val_set]
        print(f"[split] No val/valid source split; reserved {len(val_images)} train images for output val.")

    output.mkdir(parents=True)
    print(f"[classes] {categories}")
    totals: Counter = Counter()
    for name, images in (("train", train_images), ("val", val_images), ("test", splits["test"])):
        count_images, count_annotations, counters = convert_split(name, images, output, categories, valid_ids)
        totals.update(counters)
        print(f"[{name}] {count_images} images, {count_annotations} annotations -> {output / name / 'annotations.json'}")
    if totals:
        print(f"[warnings] {dict(totals)}")


if __name__ == "__main__":
    main()
