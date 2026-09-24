"""
File: src/scripts/data_download.py
Usage:
    python src/scripts/data_download.py --api-key YOUR_API_KEY
    python src/scripts/data_download.py --test-size 0.15 --seed 123
    python src/scripts/data_download.py --exclude-classes objects
Description:
    Downloads ragwort detection datasets from Roboflow Universe,
    merges them into a combined dataset with unified class mappings,
    and partitions the images and YOLO labels into train/test splits.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install via: pip install pyyaml"
    ) from exc

try:
    from roboflow import Roboflow
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'roboflow'. Install via: pip install roboflow"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "combined_dataset"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_downloads"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSpec:
    key: str  # Short prefix used in destination filenames
    workspace: str
    project: str
    version: Optional[int] = None  # None -> use latest available version


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
        "--format",
        default="yolov8",
        help="Export format supported by Roboflow, e.g. yolov8, coco (default: yolov8)",
    )
    parser.add_argument(
        "--ragwort-version",
        type=int,
        default=None,
        help="Dataset version number for ragwort-detect (default: latest)",
    )
    parser.add_argument(
        "--jkk-version",
        type=int,
        default=None,
        help="Dataset version number for jakobskreuzkraut-lsgca (default: latest)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test split (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset split (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the combined train/test dataset",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to save raw downloads before merging",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=[],
        help="Class names to omit (case-insensitive), e.g. --exclude-classes objects",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw downloaded datasets after merging",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Roboflow API key required. Pass --api-key or set ROBOFLOW_API_KEY environment variable."
        )
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be in (0, 1)")
    return args


def resolve_version(project, requested: Optional[int]):
    """Return Roboflow Version object: requested version or latest available."""
    if requested is not None:
        return project.version(requested)

    versions = project.versions()
    if not versions:
        raise RuntimeError(f"No versions found for project '{project.id}'")

    def version_number(v) -> int:
        return int(str(v.version).split("/")[-1])

    latest = max(versions, key=version_number)
    return project.version(version_number(latest))


def download_dataset(rf: Roboflow, spec: DatasetSpec, fmt: str, raw_dir: Path) -> Path:
    print(f"[download] {spec.workspace}/{spec.project} (format={fmt})...")
    project = rf.workspace(spec.workspace).project(spec.project)
    version = resolve_version(project, spec.version)

    target_dir = raw_dir / spec.key
    if target_dir.exists():
        shutil.rmtree(target_dir)

    dataset = version.download(fmt, location=str(target_dir))
    print(f"[download] saved to {dataset.location}")
    return Path(dataset.location)


def load_class_names(dataset_dir: Path) -> List[str]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    if not isinstance(names, list):
        raise ValueError(f"Unexpected 'names' format in {yaml_path}: {names!r}")
    return [str(n) for n in names]


def collect_samples(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Collect image/label pairs from train/valid/test splits."""
    samples: List[Tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = labels_dir / (image_path.stem + ".txt")
            samples.append((image_path, label_path))
    return samples


CLASS_ALIASES: Dict[str, str] = {
    "jakobskreuzkraut": "ragwort",
    "starzec": "ragwort",
}


def build_global_classes(
    per_dataset_names: Dict[str, List[str]],
    exclude: List[str],
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, Dict[int, Optional[int]]]]:
    """
    Merge class names into a unified global list with synonym aliasing
    and return local_index -> global_index mappings.
    """
    alias_map = aliases if aliases is not None else CLASS_ALIASES
    exclude_lower = {c.lower() for c in exclude}
    global_names: List[str] = []
    global_lookup: Dict[str, int] = {}
    mapping: Dict[str, Dict[int, Optional[int]]] = {}

    for key, names in per_dataset_names.items():
        mapping[key] = {}
        for local_idx, name in enumerate(names):
            canonical = alias_map.get(name.lower(), name)
            if canonical.lower() in exclude_lower or name.lower() in exclude_lower:
                mapping[key][local_idx] = None
                continue
            lookup_key = canonical.lower()
            if lookup_key not in global_lookup:
                global_lookup[lookup_key] = len(global_names)
                global_names.append(canonical)
            mapping[key][local_idx] = global_lookup[lookup_key]

    return global_names, mapping


def remap_label_file(src_label: Path, class_map: Dict[int, Optional[int]]) -> List[str]:
    """Read YOLO label file and return lines with remapped class indices."""
    if not src_label.exists():
        return []

    out_lines: List[str] = []
    with open(src_label, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            local_cls = int(parts[0])
            global_cls = class_map.get(local_cls)
            if global_cls is None:
                continue
            parts[0] = str(global_cls)
            out_lines.append(" ".join(parts))
    return out_lines


def write_split(
    samples: List[Tuple[str, Path, Path]],
    class_maps: Dict[str, Dict[int, Optional[int]]],
    split_name: str,
    output_dir: Path,
) -> int:
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for dataset_key, image_path, label_path in samples:
        out_name = f"{dataset_key}_{image_path.name}"
        shutil.copy2(image_path, images_out / out_name)

        remapped = remap_label_file(label_path, class_maps[dataset_key])
        label_out_path = labels_out / f"{dataset_key}_{image_path.stem}.txt"
        label_out_path.write_text(
            "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8"
        )
        written += 1
    return written


def write_data_yaml(output_dir: Path, class_names: List[str]) -> None:
    content = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    DATASETS[0].version = args.ragwort_version
    DATASETS[1].version = args.jkk_version

    rf = Roboflow(api_key=args.api_key)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download datasets from Roboflow Universe
    dataset_dirs: Dict[str, Path] = {}
    for spec in DATASETS:
        dataset_dirs[spec.key] = download_dataset(rf, spec, args.format, args.raw_dir)

    # 2. Build unified class mapping
    per_dataset_names = {key: load_class_names(d) for key, d in dataset_dirs.items()}
    global_classes, class_maps = build_global_classes(per_dataset_names, args.exclude_classes)
    print(f"[classes] Combined {len(global_classes)} classes: {global_classes}")

    # 3. Collect samples across datasets
    all_samples: List[Tuple[str, Path, Path]] = []
    for key, d in dataset_dirs.items():
        pairs = collect_samples(d)
        print(f"[data] {key}: {len(pairs)} images")
        all_samples.extend((key, img, lbl) for img, lbl in pairs)

    if not all_samples:
        raise RuntimeError("No images collected from datasets; aborting.")

    # 4. Partition into train/test splits
    random.shuffle(all_samples)
    n_test = max(1, round(len(all_samples) * args.test_size))
    test_samples = all_samples[:n_test]
    train_samples = all_samples[n_test:]

    n_train_written = write_split(train_samples, class_maps, "train", args.output_dir)
    n_test_written = write_split(test_samples, class_maps, "test", args.output_dir)
    write_data_yaml(args.output_dir, global_classes)

    print(f"[done] train: {n_train_written} images, test: {n_test_written} images")
    print(f"[done] Dataset written to: {args.output_dir}")
    print(f"[done] Config file: {args.output_dir / 'data.yaml'}")

    if not args.keep_raw:
        print(f"[cleanup] Removing raw downloaded data from {args.raw_dir}")
        shutil.rmtree(args.raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()