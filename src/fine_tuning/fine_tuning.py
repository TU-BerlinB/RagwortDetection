"""
File: src/fine_tuning/fine_tuning.py
Usage:
    python src/fine_tuning/fine_tuning.py --model outputs/best_model/best.pth --stage1-epochs 10 --stage2-epochs 10
    # To run validation only:
    python src/fine_tuning/fine_tuning.py --model outputs/best_model/best.pth --val-only
Description:
    Two-stage fine-tuning pipeline for ragwort detection:
      1. Stage 1 — Pure real data:
         Uses real images from data/facility (negatives with empty .txt labels)
         and data/data_concatenated/Felix (positive ragwort annotations) with AdamW.
      2. Stage 2 — Weighted real data:
         Starts from Stage 1 best checkpoint, adds a small pool of synthetic samples,
         and applies heavy oversampling (~12x) to Felix images so real features dominate gradients.
    Evaluates baseline, stage 1, and stage 2 checkpoints on a common validation set
    and saves metric comparison reports (JSON, CSV, summary).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
FINE_RESULTS_DIR = DEFAULT_OUTPUTS_DIR / "fine_labeling_results"
DEFAULT_DATASETS_DIR = FINE_RESULTS_DIR / "datasets"
STAGE1_DIR = FINE_RESULTS_DIR / "stage1"
STAGE2_DIR = FINE_RESULTS_DIR / "stage2"

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str((DEFAULT_OUTPUTS_DIR / "weights").resolve()),
        "runs_dir": str((FINE_RESULTS_DIR / "runs").resolve()),
    })
except Exception:
    pass

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_model_path(path_str: Optional[Union[str, Path]] = None) -> Path:
    """
    Find and normalize the path to baseline model weights (best.pth or best.pt).
    Automatically checks both .pth and .pt file extensions.
    """
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            candidates = [
                REPO_ROOT / p,
                DEFAULT_OUTPUTS_DIR / p,
                DEFAULT_OUTPUTS_DIR / "best_model" / p.name,
                DEFAULT_OUTPUTS_DIR / "weights" / p.name,
                DEFAULT_OUTPUTS_DIR / "models" / p.name,
            ]
        else:
            candidates = [p]
    else:
        candidates = [
            DEFAULT_OUTPUTS_DIR / "best_model" / "best.pth",
            DEFAULT_OUTPUTS_DIR / "best_model" / "best.pt",
            DEFAULT_OUTPUTS_DIR / "weights" / "best.pt",
            DEFAULT_OUTPUTS_DIR / "models" / "ragwort_yolov8_best.pt",
        ]

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
        alt_cand = cand.with_suffix(".pt") if cand.suffix.lower() == ".pth" else cand.with_suffix(".pth")
        if alt_cand.is_file():
            return alt_cand.resolve()

    raise FileNotFoundError(
        f"Baseline model weights not found. Searched locations: {[str(c) for c in candidates]}"
    )


def inspect_model_architecture(model_path: Path) -> Dict[str, Any]:
    """
    Inspect checkpoint weights structure to determine architecture (Ultralytics YOLO vs DEIM vs other).
    """
    print(f"\n[Inspect] Inspecting model architecture in: {model_path}")
    try:
        data = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Error loading weights file {model_path}: {e}")

    arch_info: Dict[str, Any] = {
        "framework": "unknown",
        "model_type": None,
        "classes": {0: "ragwort", 1: "objects"},
        "imgsz": 640,
    }

    if isinstance(data, dict):
        if "model" in data and hasattr(data["model"], "yaml"):
            arch_info["framework"] = "ultralytics_yolo"
            m = data["model"]
            arch_info["model_type"] = type(m).__name__
            if hasattr(m, "names") and m.names:
                arch_info["classes"] = m.names
            if "train_args" in data and isinstance(data["train_args"], dict):
                arch_info["imgsz"] = data["train_args"].get("imgsz", 640)
            print(f"[Inspect] Recognized Ultralytics YOLO model: {arch_info['model_type']}")
            print(f"[Inspect] Model classes: {arch_info['classes']}, imgsz: {arch_info['imgsz']}")
            return arch_info

        elif "model_state_dict" in data and "config" in data:
            arch_info["framework"] = "deim"
            arch_info["model_type"] = "DEIMModel"
            cfg = data.get("config", {})
            arch_info["classes"] = {c.get("id", 1): c.get("name", "ragwort") for c in data.get("categories", [])}
            print(f"[Inspect] Recognized DEIM / DINOv3 model: {arch_info['model_type']}")
            return arch_info

    arch_info["framework"] = "ultralytics_yolo"
    arch_info["model_type"] = "YOLO_fallback"
    print(f"[Inspect] Fallback to Ultralytics YOLO framework for: {model_path.name}")
    return arch_info


def discover_facility_images(facility_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find images in facility directory and match them with empty negative label files.
    Returns a list of (image_path, label_path) tuples.
    """
    if not facility_dir.exists():
        raise FileNotFoundError(f"Directory data/facility does not exist: {facility_dir}")

    samples: List[Tuple[Path, Path]] = []
    for p in sorted(facility_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
            lbl_candidates = [
                facility_dir / f"{p.stem}.txt",
                facility_dir / "labels" / f"{p.stem}.txt",
            ]
            lbl_path = None
            for c in lbl_candidates:
                if c.is_file():
                    lbl_path = c
                    break

            if lbl_path is None:
                lbl_path = facility_dir / f"{p.stem}.txt"

            samples.append((p.resolve(), lbl_path.resolve()))

    print(f"[Datasets] data/facility: found {len(samples)} negative images (empty .txt labels)")
    return samples


def discover_felix_images(felix_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find high-quality annotated images in Felix dataset directory and match them with YOLO .txt labels.
    """
    if not felix_dir.exists():
        alt = felix_dir.parent / "Felix_data"
        if alt.exists():
            felix_dir = alt
        else:
            raise FileNotFoundError(f"Felix directory does not exist: {felix_dir}")

    image_paths: List[Path] = []
    for p in felix_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
            image_paths.append(p.resolve())

    samples: List[Tuple[Path, Path]] = []
    for img in sorted(image_paths):
        stem = img.stem
        parent = img.parent
        possible_lbls = [
            parent / f"{stem}.txt",
            parent.parent / "annotation" / f"{stem}.txt",
            parent.parent / "labels" / f"{stem}.txt",
            parent / "labels" / f"{stem}.txt",
        ]
        lbl = None
        for cand in possible_lbls:
            if cand.is_file():
                lbl = cand
                break

        if lbl is None:
            lbl = parent / f"{stem}.txt"

        samples.append((img, lbl.resolve()))

    print(f"[Datasets] Felix ({felix_dir.name}): found {len(samples)} annotated images")
    return samples


def discover_synthetic_images(
    synth_dir: Optional[Path] = None,
    limit: int = 300,
) -> List[Tuple[Path, Path]]:
    """
    Find a limited pool of synthetic images and matching labels.
    """
    candidates = [
        synth_dir,
        REPO_ROOT / "data" / "data_concatenated" / "synthetic" / "images",
        REPO_ROOT / "data" / "data_concatenated" / "synthetic_dataset_split" / "part_1" / "images",
        REPO_ROOT / "data" / "synthetic_dataset" / "images",
    ]

    chosen_dir: Optional[Path] = None
    for cand in candidates:
        if cand and cand.is_dir() and any(cand.iterdir()):
            chosen_dir = cand
            break

    if chosen_dir is None:
        print("[Datasets] Warning: Synthetic dataset directory not found. Skipping synthetic data.")
        return []

    image_paths = [p for p in chosen_dir.glob("*.jpg") if p.is_file()]
    if not image_paths:
        image_paths = [p for p in chosen_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS]

    rnd = random.Random(42)
    rnd.shuffle(image_paths)
    selected = image_paths[:limit]

    samples: List[Tuple[Path, Path]] = []
    for img in selected:
        stem = img.stem
        cand_lbls = [
            img.parent.parent / "labels" / f"{stem}.txt",
            img.parent / f"{stem}.txt",
        ]
        lbl = None
        for cl in cand_lbls:
            if cl.is_file():
                lbl = cl
                break
        if lbl is None:
            lbl = img.parent.parent / "labels" / f"{stem}.txt"
        samples.append((img.resolve(), lbl.resolve()))

    print(f"[Datasets] Synthetic data ({chosen_dir.name}): selected {len(samples)} images (limit: {limit})")
    return samples


def discover_validation_images(
    val_felix: List[Tuple[Path, Path]],
    val_facility: List[Tuple[Path, Path]],
) -> List[Path]:
    """
    Combine validation images:
      - Split Felix positive images,
      - Split facility negative images,
      - Plus test images from existing combined_dataset test split.
    """
    val_paths: List[Path] = [p for p, _ in val_felix] + [p for p, _ in val_facility]

    test_dir_candidates = [
        REPO_ROOT / "data" / "data_concatenated" / "combined_dataset" / "test" / "images",
        REPO_ROOT / "data" / "combined_dataset" / "test" / "images",
    ]
    for td in test_dir_candidates:
        if td.is_dir():
            for p in sorted(td.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
                    val_paths.append(p.resolve())
            break

    return val_paths


def prepare_fine_tuning_datasets(
    facility_dir: Path,
    felix_dir: Path,
    output_datasets_dir: Path,
    synth_dir: Optional[Path] = None,
    felix_weight_stage2: int = 12,
    facility_weight_stage2: int = 1,
    synth_limit_stage2: int = 250,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Prepare manifests and YAML configurations for Stage 1 and Stage 2 fine-tuning.
    Non-destructive: original dataset files are preserved.
    """
    output_datasets_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)

    print("\n" + "=" * 78)
    print(" PREPARING FINE-TUNING DATASETS (NON-DESTRUCTIVE)")
    print("=" * 78)

    facility_samples = discover_facility_images(facility_dir)
    felix_samples = discover_felix_images(felix_dir)

    rnd.shuffle(facility_samples)
    rnd.shuffle(felix_samples)

    n_val_felix = max(1, int(len(felix_samples) * val_ratio))
    val_felix = felix_samples[:n_val_felix]
    train_felix = felix_samples[n_val_felix:]

    n_val_facility = max(1, int(len(facility_samples) * val_ratio))
    val_facility = facility_samples[:n_val_facility]
    train_facility = facility_samples[n_val_facility:]

    val_images = discover_validation_images(val_felix, val_facility)
    synth_samples = discover_synthetic_images(synth_dir, limit=synth_limit_stage2)

    def to_line(p: Path) -> str:
        try:
            rel = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            return rel
        except ValueError:
            return str(p.resolve())

    # Stage 1 manifest: pure real data (Felix + Facility)
    stage1_lines: List[str] = []
    for img, _ in train_felix:
        stage1_lines.append(to_line(img))
    for img, _ in train_facility:
        stage1_lines.append(to_line(img))
    rnd.shuffle(stage1_lines)

    # Stage 2 manifest: weighted real data + limited synthetic samples
    stage2_lines: List[str] = []
    for img, _ in train_felix:
        line = to_line(img)
        for _ in range(felix_weight_stage2):
            stage2_lines.append(line)

    for img, _ in train_facility:
        line = to_line(img)
        for _ in range(facility_weight_stage2):
            stage2_lines.append(line)

    for img, _ in synth_samples:
        stage2_lines.append(to_line(img))

    rnd.shuffle(stage2_lines)

    val_lines = [to_line(p) for p in val_images]

    stage1_txt = output_datasets_dir / "stage1_train.txt"
    stage2_txt = output_datasets_dir / "stage2_train.txt"
    val_txt = output_datasets_dir / "val.txt"

    stage1_txt.write_text("\n".join(stage1_lines) + "\n", encoding="utf-8")
    stage2_txt.write_text("\n".join(stage2_lines) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    base_data_cfg = {
        "path": str(REPO_ROOT.resolve()),
        "val": str(val_txt.relative_to(REPO_ROOT).as_posix()),
        "test": str(val_txt.relative_to(REPO_ROOT).as_posix()),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }

    cfg_s1 = dict(base_data_cfg)
    cfg_s1["train"] = str(stage1_txt.relative_to(REPO_ROOT).as_posix())
    stage1_yaml = output_datasets_dir / "stage1_data.yaml"
    with open(stage1_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg_s1, f, sort_keys=False, allow_unicode=True)

    cfg_s2 = dict(base_data_cfg)
    cfg_s2["train"] = str(stage2_txt.relative_to(REPO_ROOT).as_posix())
    stage2_yaml = output_datasets_dir / "stage2_data.yaml"
    with open(stage2_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg_s2, f, sort_keys=False, allow_unicode=True)

    n_s2_felix = len(train_felix) * felix_weight_stage2
    n_s2_facility = len(train_facility) * facility_weight_stage2
    n_s2_synth = len(synth_samples)
    total_s2 = len(stage2_lines)
    pct_felix_s2 = (n_s2_felix / total_s2 * 100) if total_s2 > 0 else 0
    pct_synth_s2 = (n_s2_synth / total_s2 * 100) if total_s2 > 0 else 0

    print(f"\n[Stage 1 Config - Real Data]")
    print(f"  - Positive (Felix):         {len(train_felix)} images")
    print(f"  - Negative (Facility):      {len(train_facility)} images")
    print(f"  - Total steps per epoch:    {len(stage1_lines)}")
    print(f"  - Config file:              {stage1_yaml}")

    print(f"\n[Stage 2 Config - Weighted Real Data]")
    print(f"  - Positive (Felix {felix_weight_stage2}x):   {len(train_felix)} unique -> {n_s2_felix} samples ({pct_felix_s2:.1f}% gradient share)")
    print(f"  - Negative (Facility {facility_weight_stage2}x): {len(train_facility)} unique -> {n_s2_facility} samples")
    print(f"  - Synthetic (1x):           {n_s2_synth} samples ({pct_synth_s2:.1f}% share)")
    print(f"  - Total steps per epoch:    {total_s2}")
    print(f"  - Config file:              {stage2_yaml}")

    print(f"\n[Validation Set]")
    print(f"  - Number of test samples:   {len(val_lines)} images")
    print(f"  - Validation manifest:      {val_txt}")
    print("=" * 78)

    return {
        "stage1_yaml": stage1_yaml,
        "stage2_yaml": stage2_yaml,
        "val_txt": val_txt,
        "stage1_txt": stage1_txt,
        "stage2_txt": stage2_txt,
    }


def evaluate_checkpoint(
    model_weight: Union[str, Path],
    data_yaml: Path,
    device: str,
    imgsz: int = 640,
    batch: int = 16,
) -> Dict[str, float]:
    """
    Validate a given checkpoint and return standard detection metrics.
    """
    from ultralytics import YOLO

    weight_p = Path(model_weight)
    if not weight_p.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weight_p}")

    print(f"[Validation] Loading checkpoint: {weight_p.name} on {device}...")
    model = YOLO(str(weight_p), task="detect")
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        split="val",
        device=device,
        verbose=False,
    )

    metrics = {
        "mAP50": float(res.box.map50) if hasattr(res, "box") else 0.0,
        "mAP50-95": float(res.box.map) if hasattr(res, "box") else 0.0,
        "precision": float(res.box.mp) if hasattr(res, "box") else 0.0,
        "recall": float(res.box.mr) if hasattr(res, "box") else 0.0,
        "fitness": float(res.fitness) if hasattr(res, "fitness") else 0.0,
    }
    return metrics


class FineTuningPipeline:
    """
    Two-stage model fine-tuning pipeline:
      Baseline -> Stage 1 (Real) -> Stage 2 (Weighted Real).
    """

    def __init__(
        self,
        baseline_model_path: Optional[Union[str, Path]] = None,
        facility_dir: Optional[Union[str, Path]] = None,
        felix_dir: Optional[Union[str, Path]] = None,
        synth_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch: int = 16,
        imgsz: int = 640,
        workers: int = 4,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.baseline_path = resolve_model_path(baseline_model_path)
        self.facility_dir = Path(facility_dir or (REPO_ROOT / "data" / "facility")).resolve()
        self.felix_dir = Path(felix_dir or (REPO_ROOT / "data" / "data_concatenated" / "Felix")).resolve()
        self.synth_dir = Path(synth_dir).resolve() if synth_dir else None
        self.output_dir = Path(output_dir or FINE_RESULTS_DIR).resolve()
        self.stage1_dir = self.output_dir / "stage1"
        self.stage2_dir = self.output_dir / "stage2"
        self.datasets_dir = self.output_dir / "datasets"

        self.batch = batch
        self.imgsz = imgsz
        self.workers = workers

        self.metrics_summary: Dict[str, Dict[str, Any]] = {}

    def prepare_data(
        self,
        felix_weight_stage2: int = 12,
        synth_limit_stage2: int = 250,
    ) -> Dict[str, Path]:
        """Prepare datasets for Stage 1 and Stage 2."""
        return prepare_fine_tuning_datasets(
            facility_dir=self.facility_dir,
            felix_dir=self.felix_dir,
            output_datasets_dir=self.datasets_dir,
            synth_dir=self.synth_dir,
            felix_weight_stage2=felix_weight_stage2,
            synth_limit_stage2=synth_limit_stage2,
        )

    def evaluate_baseline(self, val_yaml: Path) -> Dict[str, float]:
        """Evaluate baseline model before fine-tuning."""
        print("\n" + "=" * 78)
        print(" 1. BASELINE MODEL EVALUATION")
        print(f" Checkpoint: {self.baseline_path}")
        print("=" * 78)

        metrics = evaluate_checkpoint(
            model_weight=self.baseline_path,
            data_yaml=val_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )

        self.metrics_summary["baseline"] = {
            "checkpoint": str(self.baseline_path),
            "metrics": metrics,
        }
        self._print_metrics_block("BASELINE", metrics)
        return metrics

    def run_stage1(
        self,
        data_yaml: Path,
        epochs: int = 10,
        lr0: float = 0.001,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 1: Fine-tuning on pure real data (facility + felix).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" 2. FINE-TUNING STAGE 1 — REAL DATA")
        print(f" Starting model:  {self.baseline_path}")
        print(f" Epochs:          {epochs}")
        print(f" Learning rate:   lr0={lr0}, lrf={lrf}")
        print(f" Optimizer:       {optimizer}")
        print(f" Data:            {data_yaml}")
        print(f" Output dir:      {self.stage1_dir}")
        print("=" * 78)

        self.stage1_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self.baseline_path), task="detect")

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage1_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage1_dir / "run" / "weights" / "best.pt"
        if not best_trained.is_file():
            candidates = list(self.stage1_dir.rglob("best.pt"))
            if candidates:
                best_trained = candidates[0]
            else:
                raise FileNotFoundError(f"best.pt not found after Stage 1 training in: {self.stage1_dir}")

        stage1_best_pt = self.stage1_dir / "best.pt"
        stage1_best_pth = self.stage1_dir / "best.pth"
        shutil.copy2(best_trained, stage1_best_pt)
        try:
            if stage1_best_pth.exists() or stage1_best_pth.is_symlink():
                stage1_best_pth.unlink()
            stage1_best_pth.symlink_to(stage1_best_pt.name)
        except Exception:
            shutil.copy2(best_trained, stage1_best_pth)

        print(f"\n[Stage 1] Best model saved: {stage1_best_pt} and {stage1_best_pth}")

        metrics = evaluate_checkpoint(
            model_weight=stage1_best_pt,
            data_yaml=data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage1"] = {
            "checkpoint": str(stage1_best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 1 (REAL DATA)", metrics)
        return stage1_best_pt

    def run_stage2(
        self,
        stage1_weight: Path,
        data_yaml: Path,
        epochs: int = 10,
        lr0: float = 0.0005,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 2: Fine-tuning with weighted Felix data and limited synthetic data.
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" 3. FINE-TUNING STAGE 2 — WEIGHTED REAL DATA")
        print(f" Starting model:  {stage1_weight}")
        print(f" Epochs:          {epochs}")
        print(f" Learning rate:   lr0={lr0}, lrf={lrf}")
        print(f" Optimizer:       {optimizer}")
        print(f" Data:            {data_yaml}")
        print(f" Output dir:      {self.stage2_dir}")
        print("=" * 78)

        self.stage2_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(stage1_weight), task="detect")

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage2_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage2_dir / "run" / "weights" / "best.pt"
        if not best_trained.is_file():
            candidates = list(self.stage2_dir.rglob("best.pt"))
            if candidates:
                best_trained = candidates[0]
            else:
                raise FileNotFoundError(f"best.pt not found after Stage 2 training in: {self.stage2_dir}")

        stage2_best_pt = self.stage2_dir / "best.pt"
        stage2_best_pth = self.stage2_dir / "best.pth"
        shutil.copy2(best_trained, stage2_best_pt)
        try:
            if stage2_best_pth.exists() or stage2_best_pth.is_symlink():
                stage2_best_pth.unlink()
            stage2_best_pth.symlink_to(stage2_best_pt.name)
        except Exception:
            shutil.copy2(best_trained, stage2_best_pth)

        print(f"\n[Stage 2] Final model saved: {stage2_best_pt} and {stage2_best_pth}")

        try:
            models_dir = DEFAULT_OUTPUTS_DIR / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage2_best_pt, models_dir / "ragwort_finetuned_stage2_best.pt")
        except Exception:
            pass

        metrics = evaluate_checkpoint(
            model_weight=stage2_best_pt,
            data_yaml=data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage2"] = {
            "checkpoint": str(stage2_best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 2 (WEIGHTED REAL DATA)", metrics)
        return stage2_best_pt

    def save_comparison_report(self):
        """Generate comparison reports across baseline -> stage1 -> stage2."""
        json_path = self.output_dir / "metrics_comparison.json"
        csv_path = self.output_dir / "metrics_comparison.csv"
        txt_path = self.output_dir / "summary.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)

        csv_lines = ["stage,checkpoint,mAP50,mAP50-95,precision,recall,fitness"]
        for st_name in ["baseline", "stage1", "stage2"]:
            if st_name in self.metrics_summary:
                info = self.metrics_summary[st_name]
                m = info["metrics"]
                csv_lines.append(
                    f"{st_name},{info['checkpoint']},{m.get('mAP50', 0):.4f},"
                    f"{m.get('mAP50-95', 0):.4f},{m.get('precision', 0):.4f},"
                    f"{m.get('recall', 0):.4f},{m.get('fitness', 0):.4f}"
                )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        header = "=" * 82 + "\n"
        title = "             METRIC COMPARISON: BASELINE -> STAGE 1 -> STAGE 2\n"
        sep = "-" * 82 + "\n"
        col_names = f"{'Stage':<12} | {'mAP@50':<10} | {'mAP@50-95':<12} | {'Precision':<10} | {'Recall':<10} | {'Fitness':<10}\n"

        rows = []
        for st_name in ["baseline", "stage1", "stage2"]:
            if st_name in self.metrics_summary:
                m = self.metrics_summary[st_name]["metrics"]
                label = "Baseline" if st_name == "baseline" else ("Stage 1" if st_name == "stage1" else "Stage 2")
                row = (
                    f"{label:<12} | "
                    f"{m.get('mAP50', 0):<10.4f} | "
                    f"{m.get('mAP50-95', 0):<12.4f} | "
                    f"{m.get('precision', 0):<10.4f} | "
                    f"{m.get('recall', 0):<10.4f} | "
                    f"{m.get('fitness', 0):<10.4f}\n"
                )
                rows.append(row)

        report_txt = header + title + header + col_names + sep + "".join(rows) + header
        txt_path.write_text(report_txt, encoding="utf-8")

        print("\n" + report_txt)
        print(f"[Report] Results saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV:  {csv_path}")
        print(f"  - TXT:  {txt_path}")

    def _print_metrics_block(self, title: str, metrics: Dict[str, float]):
        print(f"\n--- VALIDATION RESULTS: {title} ---")
        print(f"  mAP@50               : {metrics.get('mAP50', 0.0):.4f}")
        print(f"  mAP@50-95            : {metrics.get('mAP50-95', 0.0):.4f}")
        print(f"  Precision            : {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall               : {metrics.get('recall', 0.0):.4f}")
        print(f"  Fitness              : {metrics.get('fitness', 0.0):.4f}")
        print("---------------------------------------")


def run_fine_tuning(
    model_path: Optional[Union[str, Path]] = None,
    facility_dir: Optional[Union[str, Path]] = None,
    felix_dir: Optional[Union[str, Path]] = None,
    synth_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    stage1_epochs: int = 10,
    stage2_epochs: int = 10,
    lr0_stage1: float = 0.001,
    lr0_stage2: float = 0.0005,
    felix_weight_stage2: int = 12,
    synth_limit_stage2: int = 250,
    batch: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None,
    workers: int = 4,
    skip_stage1: bool = False,
    stage1_checkpoint: Optional[Union[str, Path]] = None,
    val_only: bool = False,
):
    """Main execution function for the two-stage fine-tuning process."""
    resolved_model = resolve_model_path(model_path)
    inspect_model_architecture(resolved_model)

    pipeline = FineTuningPipeline(
        baseline_model_path=resolved_model,
        facility_dir=facility_dir,
        felix_dir=felix_dir,
        synth_dir=synth_dir,
        output_dir=output_dir,
        device=device,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
    )

    datasets = pipeline.prepare_data(
        felix_weight_stage2=felix_weight_stage2,
        synth_limit_stage2=synth_limit_stage2,
    )

    pipeline.evaluate_baseline(val_yaml=datasets["stage1_yaml"])
    if val_only:
        print("[Info] Ran in --val-only mode. Completed after baseline evaluation.")
        return pipeline.metrics_summary

    if skip_stage1:
        if stage1_checkpoint and Path(stage1_checkpoint).is_file():
            best_s1 = Path(stage1_checkpoint).resolve()
        else:
            best_s1 = pipeline.stage1_dir / "best.pt"
            if not best_s1.is_file():
                raise FileNotFoundError(f"--skip-stage1 active, but weights not found in: {best_s1}")
        print(f"\n[Info] Skipping Stage 1 training. Using existing checkpoint: {best_s1}")
        metrics_s1 = evaluate_checkpoint(
            best_s1, datasets["stage1_yaml"], device=pipeline.device, imgsz=imgsz, batch=batch
        )
        pipeline.metrics_summary["stage1"] = {"checkpoint": str(best_s1), "metrics": metrics_s1}
    else:
        best_s1 = pipeline.run_stage1(
            data_yaml=datasets["stage1_yaml"],
            epochs=stage1_epochs,
            lr0=lr0_stage1,
        )

    best_s2 = pipeline.run_stage2(
        stage1_weight=best_s1,
        data_yaml=datasets["stage2_yaml"],
        epochs=stage2_epochs,
        lr0=lr0_stage2,
    )

    pipeline.save_comparison_report()
    return pipeline.metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage fine-tuning pipeline for RagwortDetection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/best_model/best.pth",
        help="Path to baseline model weights (e.g. best.pth or best.pt)",
    )
    parser.add_argument(
        "--facility-dir",
        type=Path,
        default=REPO_ROOT / "data" / "facility",
        help="Path to facility negative images directory",
    )
    parser.add_argument(
        "--felix-dir",
        type=Path,
        default=REPO_ROOT / "data" / "data_concatenated" / "Felix",
        help="Path to Felix positive images directory",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=None,
        help="Path to synthetic images directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINE_RESULTS_DIR,
        help="Main output directory for fine-tuning results",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=10,
        help="Number of epochs for Stage 1 (real data)",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=10,
        help="Number of epochs for Stage 2 (weighted real data)",
    )
    parser.add_argument(
        "--lr0-stage1",
        type=float,
        default=0.001,
        help="Initial learning rate for Stage 1",
    )
    parser.add_argument(
        "--lr0-stage2",
        type=float,
        default=0.0005,
        help="Initial learning rate for Stage 2",
    )
    parser.add_argument(
        "--felix-weight",
        type=int,
        default=12,
        help="Oversampling multiplier for Felix images in Stage 2",
    )
    parser.add_argument(
        "--synth-limit",
        type=int,
        default=250,
        help="Maximum synthetic images included in Stage 2",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('0', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 and proceed to Stage 2 with existing checkpoint",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint when using --skip-stage1",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Run baseline evaluation only without training",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_fine_tuning(
        model_path=args.model,
        facility_dir=args.facility_dir,
        felix_dir=args.felix_dir,
        synth_dir=args.synth_dir,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lr0_stage1=args.lr0_stage1,
        lr0_stage2=args.lr0_stage2,
        felix_weight_stage2=args.felix_weight,
        synth_limit_stage2=args.synth_limit,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        skip_stage1=args.skip_stage1,
        stage1_checkpoint=args.stage1_checkpoint,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()