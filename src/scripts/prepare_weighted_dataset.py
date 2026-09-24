"""
File: src/scripts/prepare_weighted_dataset.py
Usage:
    python src/scripts/prepare_weighted_dataset.py
    python src/scripts/prepare_weighted_dataset.py --good-weight 15 --other-weight 1
    python src/scripts/prepare_weighted_dataset.py --no-zip
Description:
    Assembles a weighted training dataset from 'good' (high quality) and 'other' image directories,
    resolves corresponding YOLO .txt labels, applies weighted resampling multipliers,
    and exports a portable dataset package with relative paths (and optional ZIP archive).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_label_file(img_name: str, search_dirs: List[Path]) -> Optional[Path]:
    """Search for matching .txt label file for an image across candidate directories."""
    stem = Path(img_name).stem
    txt_name = f"{stem}.txt"

    for d in search_dirs:
        if not d.exists():
            continue
        candidate = d / txt_name
        if candidate.is_file():
            return candidate
        sub_candidate = d / "labels" / txt_name
        if sub_candidate.is_file():
            return sub_candidate
        for found in d.glob(f"**/{txt_name}"):
            if found.is_file():
                return found

    return None


def prepare_dataset(
    good_dir: Path,
    other_dir: Path,
    output_dir: Path,
    good_weight: int = 15,
    other_weight: int = 1,
    val_ratio: float = 0.15,
    class_names: Optional[List[str]] = None,
    make_zip: bool = True,
) -> Path:
    if class_names is None:
        class_names = ["ragwort", "objects"]

    good_dir = Path(good_dir).resolve()
    other_dir = Path(other_dir).resolve()
    output_dir = Path(output_dir).resolve()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    good_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    good_images = [p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    other_images = [p for p in other_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not good_images and not other_images:
        print("\n[!] Input directories are empty. Created directories:")
        print(f"    - GOOD (ideal images):  {good_dir}")
        print(f"    - OTHER (remaining):    {other_dir}")
        print("Place images there and re-run this script.\n")
        return output_dir

    print("=" * 70)
    print(" PREPARING WEIGHTED DATASET FOR EXPORT")
    print("=" * 70)
    print(f"Directory GOOD:   {good_dir} ({len(good_images)} images, weight = {good_weight}x)")
    print(f"Directory OTHER:  {other_dir} ({len(other_images)} images, weight = {other_weight}x)")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    label_search_dirs = [
        good_dir,
        other_dir,
        REPO_ROOT / "data" / "combined_dataset" / "train" / "labels",
        REPO_ROOT / "data" / "combined_dataset" / "test" / "labels",
        REPO_ROOT / "data" / "ragwort_segmentation" / "train" / "labels",
        REPO_ROOT / "data" / "ragwort_segmentation" / "val" / "labels",
        REPO_ROOT / "data",
    ]

    random.seed(42)

    def split_set(items: List[Path], ratio: float) -> Tuple[List[Path], List[Path]]:
        shuffled = list(items)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * ratio)) if len(shuffled) > 3 else 0
        return shuffled[n_val:], shuffled[:n_val]

    train_good, val_good = split_set(good_images, val_ratio)
    train_other, val_other = split_set(other_images, val_ratio)

    train_all_images = [(p, "good", good_weight) for p in train_good] + [
        (p, "other", other_weight) for p in train_other
    ]
    val_all_images = [(p, "good") for p in val_good] + [(p, "other") for p in val_other]

    out_train_img = output_dir / "images" / "train"
    out_val_img = output_dir / "images" / "val"
    out_train_lbl = output_dir / "labels" / "train"
    out_val_lbl = output_dir / "labels" / "val"

    for d in [out_train_img, out_val_img, out_train_lbl, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_file_or_link(src: Path, dst: Path):
        if not dst.exists():
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    missing_labels = []
    train_txt_relative_lines: List[str] = []

    for img_path, category, weight in train_all_images:
        dest_img = out_train_img / img_path.name
        copy_file_or_link(img_path, dest_img)

        lbl_file = find_label_file(img_path.name, label_search_dirs)
        dest_lbl = out_train_lbl / f"{img_path.stem}.txt"
        if lbl_file and lbl_file.exists():
            copy_file_or_link(lbl_file, dest_lbl)
        else:
            dest_lbl.write_text("", encoding="utf-8")
            missing_labels.append(img_path.name)

        rel_img_path = f"images/train/{img_path.name}"
        for _ in range(weight):
            train_txt_relative_lines.append(rel_img_path)

    val_txt_relative_lines: List[str] = []
    for img_path, category in val_all_images:
        dest_img = out_val_img / img_path.name
        copy_file_or_link(img_path, dest_img)

        lbl_file = find_label_file(img_path.name, label_search_dirs)
        dest_lbl = out_val_lbl / f"{img_path.stem}.txt"
        if lbl_file and lbl_file.exists():
            copy_file_or_link(lbl_file, dest_lbl)
        else:
            dest_lbl.write_text("", encoding="utf-8")

        val_txt_relative_lines.append(f"images/val/{img_path.name}")

    if missing_labels:
        print(f"[Notice] For {len(missing_labels)} images no .txt label was found (saved as empty negative background).")

    manifest_train = output_dir / "train_weighted.txt"
    with open(manifest_train, "w", encoding="utf-8") as f:
        f.write("\n".join(train_txt_relative_lines) + "\n")

    manifest_val = output_dir / "val.txt"
    with open(manifest_val, "w", encoding="utf-8") as f:
        f.write("\n".join(val_txt_relative_lines) + "\n")

    data_yaml_content = {
        "path": ".",
        "train": "train_weighted.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, allow_unicode=True)

    train_script_content = '''"""
train.py - Standalone training script for YOLOv8 on exported dataset.
Usage:
    python train.py
"""

import torch
from ultralytics import YOLO

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device} (CUDA: {torch.cuda.is_available()})")

    model = YOLO("yolov8s.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        save=True,
        project="runs_ragwort",
        name="yolov8s_weighted",
        exist_ok=True,
    )
    print("Training finished.")

if __name__ == "__main__":
    main()
'''
    with open(output_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(train_script_content)

    bat_content = "@echo off\npip install ultralytics torch torchvision\npython train.py\npause\n"
    with open(output_dir / "run_train.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)

    sh_content = "#!/bin/bash\npip install ultralytics torch torchvision\npython train.py\n"
    with open(output_dir / "run_train.sh", "w", encoding="utf-8") as f:
        f.write(sh_content)

    total_good_train = len(train_good)
    total_other_train = len(train_other)
    weighted_good_samples = total_good_train * good_weight
    weighted_other_samples = total_other_train * other_weight
    total_samples = weighted_good_samples + weighted_other_samples
    pct_good = (weighted_good_samples / total_samples * 100) if total_samples > 0 else 0

    print("\n" + "=" * 70)
    print(" DATASET EXPORT READY")
    print("=" * 70)
    print(f"Training:")
    print(f"  - Unique GOOD images:  {total_good_train} (weight {good_weight}x -> {weighted_good_samples} samples)")
    print(f"  - Unique OTHER images: {total_other_train} (weight {other_weight}x -> {weighted_other_samples} samples)")
    print(f"  - Total steps/epoch:   {total_samples}")
    print(f"  - GOOD images share:   {pct_good:.1f}%")
    print(f"Validation:")
    print(f"  - Test images:         {len(val_all_images)}")
    print("=" * 70)

    if make_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        print(f"\n[Packaging] Creating ZIP archive: {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_p = Path(root) / file
                    rel_p = file_p.relative_to(output_dir.parent)
                    zf.write(file_p, arcname=str(rel_p))
        print(f"[Packaging] Finished ZIP archive: {zip_path}")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a weighted standalone dataset from 'good' and 'other' image folders."
    )
    parser.add_argument(
        "--good-dir",
        type=Path,
        default=REPO_ROOT / "dataset_input" / "good",
        help="Directory with ideal images (default: dataset_input/good)",
    )
    parser.add_argument(
        "--other-dir",
        type=Path,
        default=REPO_ROOT / "dataset_input" / "other",
        help="Directory with other images (default: dataset_input/other)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dataset_export_ready",
        help="Target export directory",
    )
    parser.add_argument(
        "--good-weight",
        type=int,
        default=15,
        help="Oversampling multiplier for good images (default: 15x)",
    )
    parser.add_argument(
        "--other-weight",
        type=int,
        default=1,
        help="Multiplier for other images (default: 1x)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create ZIP archive",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    good_dir = args.good_dir
    other_dir = args.other_dir

    if (REPO_ROOT / "good").exists() and any((REPO_ROOT / "good").iterdir()):
        good_dir = REPO_ROOT / "good"
    if (REPO_ROOT / "other").exists() and any((REPO_ROOT / "other").iterdir()):
        other_dir = REPO_ROOT / "other"

    prepare_dataset(
        good_dir=good_dir,
        other_dir=other_dir,
        output_dir=args.output_dir,
        good_weight=args.good_weight,
        other_weight=args.other_weight,
        make_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()
