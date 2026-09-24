"""
File: src/scripts/prepare_dino_data.py
Usage:
    python src/scripts/prepare_dino_data.py --from-combined data/combined_dataset
    python src/scripts/prepare_dino_data.py --api-key YOUR_ROBOFLOW_KEY
Description:
    Downloads or processes Roboflow/YOLO datasets into a binary classification
    directory hierarchy (data/dino_dataset/ragwort and data/dino_dataset/others),
    directly compatible with torchvision.datasets.ImageFolder and DINOv3 embeddings.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dino_dataset"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_downloads"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RAGWORT_SYNONYMS = {"ragwort", "jakobskreuzkraut", "starzec", "senecio"}


@dataclass
class DatasetSpec:
    key: str
    workspace: str
    project: str
    version: Optional[int] = None


DATASETS: List[DatasetSpec] = [
    DatasetSpec(key="ragwort", workspace="group-project-i4pjs", project="ragwort-detect"),
    DatasetSpec(key="jkk", workspace="jakobskreuzkraut", project="jakobskreuzkraut-lsgca"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY environment variable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory with 'ragwort' and 'others' subfolders (default: data/dino_dataset)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw downloads from Roboflow",
    )
    parser.add_argument(
        "--from-combined",
        type=Path,
        default=None,
        help="Optional path to existing combined_dataset (bypasses Roboflow re-download)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw downloaded data after splitting",
    )
    return parser.parse_args()


def load_class_names(data_yaml_path: Path) -> List[str]:
    """Load class names list from data.yaml."""
    if not data_yaml_path.exists():
        return []
    with open(data_yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    return [str(n) for n in names]


def resolve_version(project, requested: Optional[int]):
    """Return requested or latest project Version object."""
    if requested is not None:
        return project.version(requested)
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"No versions found for project '{project.id}'")

    def version_number(v) -> int:
        return int(str(v.version).split("/")[-1])

    latest = max(versions, key=version_number)
    return project.version(version_number(latest))


def download_raw_datasets(rf: Roboflow, raw_dir: Path) -> Dict[str, Path]:
    """Download datasets to raw_dir."""
    dataset_dirs = {}
    for spec in DATASETS:
        target_dir = raw_dir / spec.key
        if target_dir.exists():
            shutil.rmtree(target_dir)

        print(f"[download] {spec.workspace}/{spec.project}...")
        project = rf.workspace(spec.workspace).project(spec.project)
        version = resolve_version(project, spec.version)
        dataset = version.download("yolov8", location=str(target_dir))
        dataset_dirs[spec.key] = Path(dataset.location)
    return dataset_dirs


def has_ragwort_label(label_path: Path, ragwort_class_ids: Set[int]) -> bool:
    """Check if label file contains at least one bounding box with a ragwort class ID."""
    if not label_path.exists():
        return False

    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id in ragwort_class_ids:
                return True
    return False


def collect_images_and_labels(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Collect (image_path, label_path) pairs across splits in dataset_dir."""
    pairs = []
    splits_to_check = [
        dataset_dir / "train",
        dataset_dir / "valid",
        dataset_dir / "test",
        dataset_dir,
    ]
    for sp in splits_to_check:
        images_dir = sp / "images" if (sp / "images").exists() else (sp if sp.name == "images" else None)
        if not images_dir or not images_dir.exists():
            continue
        labels_dir = images_dir.parent / "labels"

        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
                lbl = labels_dir / (img.stem + ".txt")
                pairs.append((img, lbl))
    return pairs


def split_into_classification(
    dataset_dirs: Dict[str, Path],
    output_dir: Path,
) -> Tuple[int, int]:
    """Copy images into output_dir / 'ragwort' and output_dir / 'others' based on annotations."""
    ragwort_out = output_dir / "ragwort"
    others_out = output_dir / "others"

    ragwort_out.mkdir(parents=True, exist_ok=True)
    others_out.mkdir(parents=True, exist_ok=True)

    ragwort_count = 0
    others_count = 0

    for dataset_key, d_dir in dataset_dirs.items():
        class_names = load_class_names(d_dir / "data.yaml")
        ragwort_ids: Set[int] = set()

        for idx, name in enumerate(class_names):
            if name.lower() in RAGWORT_SYNONYMS:
                ragwort_ids.add(idx)

        if not ragwort_ids and dataset_key == "ragwort":
            ragwort_ids.add(0)

        pairs = collect_images_and_labels(d_dir)
        print(f"[processing] Dataset '{dataset_key}': {len(pairs)} images (ragwort IDs: {ragwort_ids})")

        for img_path, lbl_path in pairs:
            has_rag = has_ragwort_label(lbl_path, ragwort_ids)
            out_filename = f"{dataset_key}_{img_path.name}"

            if has_rag:
                shutil.copy2(img_path, ragwort_out / out_filename)
                ragwort_count += 1
            else:
                shutil.copy2(img_path, others_out / out_filename)
                others_count += 1

    return ragwort_count, others_count


def split_from_combined_dataset(combined_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """Partition pre-combined dataset into ragwort and others directories."""
    ragwort_out = output_dir / "ragwort"
    others_out = output_dir / "others"

    ragwort_out.mkdir(parents=True, exist_ok=True)
    others_out.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(combined_dir / "data.yaml")
    ragwort_ids = {idx for idx, name in enumerate(class_names) if name.lower() in RAGWORT_SYNONYMS}
    if not ragwort_ids:
        ragwort_ids.add(0)

    pairs = collect_images_and_labels(combined_dir)
    print(f"[processing] Combined dataset ({combined_dir}): {len(pairs)} images (ragwort IDs: {ragwort_ids})")

    ragwort_count = 0
    others_count = 0

    for img_path, lbl_path in pairs:
        has_rag = has_ragwort_label(lbl_path, ragwort_ids)
        out_filename = img_path.name

        if has_rag:
            shutil.copy2(img_path, ragwort_out / out_filename)
            ragwort_count += 1
        else:
            shutil.copy2(img_path, others_out / out_filename)
            others_count += 1

    return ragwort_count, others_count


def main():
    args = parse_args()
    print("=== DINO Classification Dataset Preparation ===")
    print(f"Output directory: {args.output_dir}")

    if args.from_combined and args.from_combined.exists():
        print(f"[INFO] Using existing combined dataset: {args.from_combined}")
        n_rag, n_oth = split_from_combined_dataset(args.from_combined, args.output_dir)

    elif (REPO_ROOT / "data" / "combined_dataset").exists() and any((REPO_ROOT / "data" / "combined_dataset").iterdir()):
        print("[INFO] Found existing data in data/combined_dataset.")
        n_rag, n_oth = split_from_combined_dataset(REPO_ROOT / "data" / "combined_dataset", args.output_dir)

    elif (args.raw_dir / "ragwort").exists() and (args.raw_dir / "jkk").exists():
        print(f"[INFO] Found downloaded data in {args.raw_dir}.")
        dataset_dirs = {"ragwort": args.raw_dir / "ragwort", "jkk": args.raw_dir / "jkk"}
        n_rag, n_oth = split_into_classification(dataset_dirs, args.output_dir)

    else:
        if not args.api_key:
            raise SystemExit(
                "Error: Roboflow API key required to download datasets.\n"
                "Specify --api-key YOUR_KEY or set ROBOFLOW_API_KEY environment variable,\n"
                "or specify existing data via --from-combined."
            )
        if not ROBOFLOW_AVAILABLE:
            raise SystemExit("Error: Missing 'roboflow' package. Install via: pip install roboflow")

        rf = Roboflow(api_key=args.api_key)
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        dataset_dirs = download_raw_datasets(rf, args.raw_dir)
        n_rag, n_oth = split_into_classification(dataset_dirs, args.output_dir)

        if not args.keep_raw:
            print(f"[cleanup] Removing raw downloaded data from {args.raw_dir}")
            shutil.rmtree(args.raw_dir, ignore_errors=True)

    print("\n--- Summary ---")
    print(f"Ragwort images: {n_rag} -> {args.output_dir / 'ragwort'}")
    print(f"Other images:   {n_oth} -> {args.output_dir / 'others'}")


if __name__ == "__main__":
    main()
