"""
File: src/scripts/select_representative.py
Usage:
    python src/scripts/select_representative.py --dry-run
    python src/scripts/select_representative.py --n-select 150
    python src/scripts/select_representative.py --split all --extra-dir data/plants --n-select 200 --equal-classes
    python src/scripts/select_representative.py --n-select 150 --copy-images
Description:
    Selects a representative subset of images for annotation using facility location
    (greedy k-center) algorithm applied to DINOv3 / DINOv2 vision transformer embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "combined_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RAGWORT_SYNONYMS = {"ragwort", "jakobskreuzkraut", "starzec", "senecio"}

CLASS_RAGWORT = "ragwort"
CLASS_OTHERS = "others"

MODEL_CANDIDATES = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov2-small",
]

HF_HINT = (
    "Model DINOv3 is gated. To access it:\n"
    "  1) Visit https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
    "     and click 'Agree and access repository'\n"
    "  2) Authenticate: pip install -U huggingface_hub && hf auth login\n"
    "     (or: export HF_TOKEN=hf_your_token)"
)


@dataclass
class Sample:
    path: Path
    label: str

    @property
    def rel_path(self) -> str:
        try:
            return self.path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


def load_ragwort_class_ids(data_dir: Path) -> Set[int]:
    yaml_path = data_dir / "data.yaml"
    names: List[str] = []

    if yaml_path.exists():
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw = data.get("names", [])
            if isinstance(raw, dict):
                raw = [raw[k] for k in sorted(raw, key=int)]
            names = [str(n) for n in raw]
        except ImportError:
            print("[notice] pyyaml not found; assuming class 0 is ragwort")
        except Exception as exc:
            print(f"[notice] failed loading {yaml_path}: {exc}")

    ids = {i for i, n in enumerate(names) if n.strip().lower() in RAGWORT_SYNONYMS}
    return ids or {0}


def label_of_image(label_path: Path, ragwort_ids: Set[int]) -> str:
    if not label_path.exists():
        return CLASS_OTHERS

    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                if int(float(parts[0])) in ragwort_ids:
                    return CLASS_RAGWORT
            except ValueError:
                continue
    return CLASS_OTHERS


def collect_samples(data_dir: Path, splits: Sequence[str]) -> List[Sample]:
    ragwort_ids = load_ragwort_class_ids(data_dir)
    samples: List[Sample] = []

    for split in splits:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"

        if not images_dir.exists():
            print(f"[notice] Skipping split '{split}' - missing {images_dir}")
            continue

        paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for img in paths:
            samples.append(Sample(img, label_of_image(labels_dir / (img.stem + ".txt"), ragwort_ids)))
        print(f"[scan] {split}: {len(paths)} images (ragwort class IDs: {sorted(ragwort_ids)})")

    return samples


def collect_extra_dir(extra_dir: Path, label: str) -> List[Sample]:
    if not extra_dir.exists():
        print(f"[notice] Skipping --extra-dir - missing {extra_dir}")
        return []

    paths = sorted(p for p in extra_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"[scan] {extra_dir}: {len(paths)} images as '{label}'")
    return [Sample(p, label) for p in paths]


def subsample(samples: List[Sample], limit: int) -> List[Sample]:
    if limit <= 0 or limit >= len(samples):
        return samples

    step = len(samples) / limit
    picked = [samples[min(len(samples) - 1, int(i * step))] for i in range(limit)]
    print(f"[limit] Selecting {len(picked)} of {len(samples)} images")
    return picked


def load_backbone(model_name: str | None):
    try:
        from src.models.dinov3 import DINOv3
    except ImportError:
        from models.dinov3 import DINOv3

    if model_name:
        return DINOv3(model_name=model_name), model_name

    last_error = None
    for candidate in MODEL_CANDIDATES:
        try:
            print(f"[model] Trying: {candidate}")
            return DINOv3(model_name=candidate), candidate
        except Exception as exc:
            last_error = exc
            print(f"[model] {candidate} unavailable ({type(exc).__name__}) -> trying fallback")
            if candidate.startswith("facebook/dinov3"):
                print(HF_HINT)

    raise SystemExit(f"Error: failed loading vision backbone. Last error: {last_error}")


def compute_embeddings(samples: List[Sample], batch_size: int, model_name: str | None) -> Tuple[np.ndarray, str]:
    import torch
    from PIL import Image
    from tqdm import tqdm

    model, used_name = load_backbone(model_name)
    print(f"[model] Using {used_name} on: {model.device}")

    vectors: List[np.ndarray] = []
    for start in tqdm(range(0, len(samples), batch_size), desc="Embeddings"):
        batch = samples[start : start + batch_size]
        images = [Image.open(s.path).convert("RGB") for s in batch]

        with torch.no_grad():
            features = model.extract_features(images)

        vectors.append(features.mean(dim=1).cpu().numpy())
        for img in images:
            img.close()

    return np.vstack(vectors).astype(np.float32), used_name


def load_or_compute_embeddings(
    samples: List[Sample], cache_path: Path, batch_size: int, recompute: bool, model_name: str | None
) -> Tuple[np.ndarray, str]:
    current = np.array([s.rel_path for s in samples])

    if cache_path.exists() and not recompute:
        cached = np.load(cache_path, allow_pickle=False)
        cached_model = str(cached["model"]) if "model" in cached else "?"
        same_files = "paths" in cached and len(cached["paths"]) == len(current) and np.array_equal(cached["paths"], current)
        same_model = model_name is None or cached_model == model_name

        if same_files and same_model:
            print(f"[cache] Loading embeddings from {cache_path} (model: {cached_model})")
            return cached["embeddings"].astype(np.float32), cached_model
        print("[cache] File list or model changed; recomputing embeddings")

    embeddings, used_name = compute_embeddings(samples, batch_size, model_name)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        paths=current,
        labels=np.array([s.label for s in samples]),
        model=np.array(used_name),
    )
    print(f"[cache] Saved {cache_path} (shape: {embeddings.shape})")

    return embeddings, used_name


def prepare_metric_space(embeddings: np.ndarray, metric: str) -> np.ndarray:
    if metric != "cosine":
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def greedy_k_center(X: np.ndarray, p: int, start: str = "medoid", seed: int = 0) -> Tuple[List[int], List[float], List[float]]:
    n = X.shape[0]
    p = max(1, min(p, n))

    if start == "random":
        first = int(np.random.default_rng(seed).integers(n))
    else:
        first = int(np.argmin(np.linalg.norm(X - X.mean(axis=0), axis=1)))

    selected = [first]
    min_dist = np.linalg.norm(X - X[first], axis=1)
    gap_at_pick = [float("inf")]
    radius_after = [float(min_dist.max())]

    for _ in range(p - 1):
        nxt = int(np.argmax(min_dist))
        gap = float(min_dist[nxt])
        if gap <= 0.0:
            break

        selected.append(nxt)
        gap_at_pick.append(gap)
        min_dist = np.minimum(min_dist, np.linalg.norm(X - X[nxt], axis=1))
        radius_after.append(float(min_dist.max()))

    return selected, gap_at_pick, radius_after


def allocate_per_class(counts: Dict[str, int], n_select: int, equal: bool) -> Dict[str, int]:
    classes = sorted(counts)
    total = sum(counts.values())
    if total == 0:
        return {c: 0 for c in classes}

    if equal:
        base = n_select // len(classes)
        alloc = {c: min(base, counts[c]) for c in classes}
    else:
        alloc = {c: min(counts[c], int(round(n_select * counts[c] / total))) for c in classes}

    while sum(alloc.values()) < n_select and any(alloc[c] < counts[c] for c in classes):
        alloc[max(classes, key=lambda k: counts[k] - alloc[k])] += 1
    while sum(alloc.values()) > n_select:
        alloc[max(classes, key=lambda k: alloc[k])] -= 1

    return alloc


def write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "filename", "label", "pick_order", "gap_at_pick", "radius_after"])
        writer.writeheader()
        writer.writerows(rows)


def copy_selected(rows: List[dict], target_dir: Path) -> None:
    for row in rows:
        dest_dir = target_dir / row["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / row["path"], dest_dir / f"{row['pick_order']:03d}_{row['filename']}")
    print(f"[copy] Selected images copied to {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--extra-dir", type=Path, default=None, help="Additional directory of images (e.g. data/plants)")
    parser.add_argument("--extra-label", default=CLASS_OTHERS)
    parser.add_argument("--n-select", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="Use subset of N images (0 = all)")
    parser.add_argument("--model", default=None, help="e.g. facebook/dinov2-small (default: DINOv3 with DINOv2 fallback)")
    parser.add_argument("--equal-classes", action="store_true")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--start", default="medoid", choices=["medoid", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]

    print("=== Representative Image Selection (Facility Location / k-Center) ===")
    print(f"Data: {args.data_dir} (splits: {', '.join(splits)})")

    samples = collect_samples(args.data_dir, splits)
    if args.extra_dir is not None:
        samples += collect_extra_dir(args.extra_dir, args.extra_label)

    if not samples:
        raise SystemExit(f"Error: no images found in {args.data_dir}.")

    samples = subsample(samples, args.limit)

    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1

    print(f"[pool] {len(samples)} images: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    alloc = allocate_per_class(counts, args.n_select, args.equal_classes)
    print("[allocation] Selection targets: " + ", ".join(f"{k}={v}" for k, v in sorted(alloc.items())))

    if args.dry_run:
        print("\n[dry-run] Finished without computing embeddings or selections.")
        return

    suffix = f"_limit{args.limit}" if args.limit > 0 else ""
    cache_name = f"embeddings_combined_{'_'.join(splits)}{suffix}.npz"
    embeddings, used_model = load_or_compute_embeddings(
        samples, args.output_dir / cache_name, args.batch_size, args.recompute, args.model
    )
    X = prepare_metric_space(embeddings, args.metric)

    rows: List[dict] = []
    per_class: Dict[str, dict] = {}

    for label in sorted(counts):
        pool_idx = [i for i, s in enumerate(samples) if s.label == label]
        if alloc[label] <= 0:
            continue

        selected, gaps, radii = greedy_k_center(X[pool_idx], alloc[label], args.start, args.seed)

        for order, (local, gap, radius) in enumerate(zip(selected, gaps, radii), start=1):
            sample = samples[pool_idx[local]]
            rows.append(
                {
                    "path": sample.rel_path,
                    "filename": sample.path.name,
                    "label": label,
                    "pick_order": order,
                    "gap_at_pick": "" if gap == float("inf") else round(gap, 6),
                    "radius_after": round(radius, 6),
                }
            )

        per_class[label] = {
            "pool": counts[label],
            "selected": len(selected),
            "start_radius": round(radii[0], 6),
            "final_radius": round(radii[-1], 6),
        }
        print(f"[k-center] {label}: {len(selected)}/{counts[label]}, radius {radii[0]:.4f} -> {radii[-1]:.4f}")

    csv_path = args.output_dir / "selected_representative.csv"
    write_csv(rows, csv_path)

    summary_path = args.output_dir / "selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "splits": splits,
                "extra_dir": str(args.extra_dir) if args.extra_dir else None,
                "model": used_model,
                "metric": args.metric,
                "start": args.start,
                "limit": args.limit,
                "n_select_requested": args.n_select,
                "n_select_actual": len(rows),
                "pool_counts": counts,
                "allocation": alloc,
                "per_class": per_class,
                "embeddings_cache": str(args.output_dir / cache_name),
                "csv": str(csv_path),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    if args.copy_images:
        copy_selected(rows, args.output_dir / "selected_images")

    print(f"\nFinished. Selected images manifest: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
