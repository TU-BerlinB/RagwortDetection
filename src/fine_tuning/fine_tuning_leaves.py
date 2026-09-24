"""
File: src/fine_tuning/fine_tuning_leaves.py
Usage:
    python src/fine_tuning/fine_tuning_leaves.py --model outputs/fine_labeling_results/stage2/best.pt
    # To run validation only:
    python src/fine_tuning/fine_tuning_leaves.py --val-only
Description:
    Extended fine-tuning pipeline focusing on ragwort leaf geometry and structural features:
      - Stage 3: Increased input resolution (imgsz=1024)
      - Stage 4: Leaf-focused augmentations (random erasing of flowers, HSV jitter)
      - Stage 5: No-Yellow training with flowers desaturated to force learning leaf characteristics
    Evaluates checkpoints across Stages 2, 3, 4, and 5 on a shared validation set
    and exports comparison summary reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FINE_RESULTS_DIR = REPO_ROOT / "outputs" / "fine_labeling_results"
STAGE2_DEFAULT_MODEL = FINE_RESULTS_DIR / "stage2" / "best.pt"
DATASETS_DIR = FINE_RESULTS_DIR / "datasets"

STAGE3_DIR = FINE_RESULTS_DIR / "leaves_stage3_resolution"
STAGE4_DIR = FINE_RESULTS_DIR / "leaves_stage4_augmentation"
STAGE5_DIR = FINE_RESULTS_DIR / "leaves_stage5_no_yellow"

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str((REPO_ROOT / "outputs" / "weights").resolve()),
        "runs_dir": str((FINE_RESULTS_DIR / "runs").resolve()),
    })
except Exception:
    pass

from src.fine_tuning.fine_tuning import (
    evaluate_checkpoint,
    inspect_model_architecture,
    resolve_model_path,
)


def remove_yellow_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Perform controlled HSV color space manipulation:
    desaturates distinctive yellow flower petals while preserving natural
    green leaf tones, stems, and background colors.

    OpenCV HSV parameters:
      - Yellow flower petals: H in [18, 36], S >= 40, V >= 50
      - Green leaves and stems: H in [38, 85] (no mask overlap)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 40, 50], dtype=np.uint8)
    upper_yellow = np.array([36, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Desaturate yellow pixels so flowers appear as neutral off-white / withered grey
    h, s, v = cv2.split(hsv)
    s_modified = np.where(yellow_mask > 0, (s * 0.05).astype(np.uint8), s)
    modified_hsv = cv2.merge([h, s_modified, v])

    return cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)


def prepare_no_yellow_dataset(
    source_manifest: Path,
    output_dataset_dir: Path,
    val_manifest: Path,
) -> Path:
    """
    Generate a transformed training dataset copy with yellow color desaturated.
    Original dataset files in data/ remain completely unmodified.
    """
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dataset_dir / "images"
    labels_out = output_dataset_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(" PREPARING NO-YELLOW DATASET (STAGE 5)")
    print(f" Source manifest: {source_manifest}")
    print(f" Output dir:      {output_dataset_dir}")
    print("=" * 78)

    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")

    lines = source_manifest.read_text(encoding="utf-8").splitlines()
    unique_image_paths = sorted(list(set(l.strip() for l in lines if l.strip())))

    print(f"[No-Yellow] Processing {len(unique_image_paths)} unique images...")

    rel_to_transformed: Dict[str, str] = {}
    converted_count = 0

    for idx, line_path in enumerate(unique_image_paths):
        p = Path(line_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()

        if not p.is_file():
            continue

        out_img_name = f"noyellow_{p.stem}{p.suffix.lower()}"
        out_img_path = images_out / out_img_name
        out_lbl_path = labels_out / f"{out_img_name.rsplit('.', 1)[0]}.txt"

        if not out_img_path.is_file():
            img_bgr = cv2.imread(str(p))
            if img_bgr is not None:
                transformed_bgr = remove_yellow_from_image(img_bgr)
                cv2.imwrite(str(out_img_path), transformed_bgr)
                converted_count += 1
            else:
                shutil.copy2(p, out_img_path)

        if not out_lbl_path.is_file():
            stem = p.stem
            cand_lbls = [
                p.parent / f"{stem}.txt",
                p.parent.parent / "labels" / f"{stem}.txt",
                p.parent.parent / "annotation" / f"{stem}.txt",
                p.parent / "labels" / f"{stem}.txt",
            ]
            src_lbl = None
            for c in cand_lbls:
                if c.is_file():
                    src_lbl = c
                    break

            if src_lbl and src_lbl.is_file() and src_lbl.stat().st_size > 0:
                shutil.copy2(src_lbl, out_lbl_path)
            else:
                out_lbl_path.write_text("", encoding="utf-8")

        try:
            rel_to_repo = out_img_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel_to_repo = str(out_img_path.resolve())
        rel_to_transformed[line_path] = rel_to_repo

    print(f"[No-Yellow] Modified {converted_count} images (desaturated yellow flower pixels).")
    print(f"[No-Yellow] Annotations and image geometries preserved 1:1.")

    new_train_lines = []
    for l in lines:
        raw_l = l.strip()
        if raw_l in rel_to_transformed:
            new_train_lines.append(rel_to_transformed[raw_l])

    train_no_yellow_txt = output_dataset_dir / "train_no_yellow.txt"
    train_no_yellow_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")

    no_yellow_yaml = output_dataset_dir / "data_no_yellow.yaml"

    def get_rel_str(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT.resolve()).as_posix())
        except ValueError:
            return str(p.resolve())

    cfg = {
        "path": str(REPO_ROOT.resolve()),
        "train": get_rel_str(train_no_yellow_txt),
        "val": get_rel_str(val_manifest),
        "test": get_rel_str(val_manifest),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }
    with open(no_yellow_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[No-Yellow] Saved training manifest: {train_no_yellow_txt}")
    print(f"[No-Yellow] Saved YAML config:        {no_yellow_yaml}")
    print("=" * 78)

    return no_yellow_yaml


class LeavesFineTuningPipeline:
    """
    Extended fine-tuning pipeline focused on leaf and structural features:
      Stage 2 (Base) -> Stage 3 (1024 Res) -> Stage 4 (Augmentations) -> Stage 5 (No-Yellow)
    """

    def __init__(
        self,
        stage2_model_path: Optional[Union[str, Path]] = None,
        data_yaml: Optional[Union[str, Path]] = None,
        val_manifest: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        imgsz: int = 1024,
        batch: int = 8,
        device: Optional[str] = None,
        workers: int = 4,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir or FINE_RESULTS_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cand_stage2 = stage2_model_path or STAGE2_DEFAULT_MODEL
        self.stage2_model_path = resolve_model_path(cand_stage2)

        default_yaml = DATASETS_DIR / "stage2_data.yaml"
        self.data_yaml = Path(data_yaml or default_yaml).resolve()
        if not self.data_yaml.is_file():
            cands = list(DATASETS_DIR.glob("*data*.yaml"))
            if cands:
                self.data_yaml = cands[0].resolve()
            else:
                raise FileNotFoundError(f"YAML dataset config not found: {self.data_yaml}")

        default_val = DATASETS_DIR / "val.txt"
        self.val_manifest = Path(val_manifest or default_val).resolve()

        self.imgsz = imgsz
        self.batch = batch
        self.workers = workers

        self.stage3_dir = self.output_dir / "leaves_stage3_resolution"
        self.stage4_dir = self.output_dir / "leaves_stage4_augmentation"
        self.stage5_dir = self.output_dir / "leaves_stage5_no_yellow"

        self.metrics_summary: Dict[str, Dict[str, Any]] = {}

    def evaluate_initial_stage2(self) -> Dict[str, float]:
        """Evaluate input Stage 2 model at high resolution (1024)."""
        print("\n" + "=" * 78)
        print(" 0. BASE STAGE 2 MODEL EVALUATION")
        print(f" Checkpoint: {self.stage2_model_path}")
        print(f" Imgsz:      {self.imgsz}")
        print("=" * 78)

        metrics = evaluate_checkpoint(
            model_weight=self.stage2_model_path,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )

        self.metrics_summary["stage2"] = {
            "checkpoint": str(self.stage2_model_path),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 2 (INPUT BASE)", metrics)
        return metrics

    def run_stage3(
        self,
        epochs: int = 8,
        lr0: float = 0.0005,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 3: Increased image resolution (imgsz=1024).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 3 — INCREASED RESOLUTION (imgsz=1024)")
        print(f" Input model:     {self.stage2_model_path}")
        print(f" Epochs:          {epochs}")
        print(f" Batch / Imgsz:   {self.batch} / {self.imgsz}")
        print(f" Learning rate:   lr0={lr0}, lrf={lrf}")
        print(f" Output dir:      {self.stage3_dir}")
        print("=" * 78)

        self.stage3_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self.stage2_model_path), task="detect")

        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage3_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage3_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage3_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage3_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"best.pt weights not found in: {self.stage3_dir}")

        best_pt = self.stage3_dir / "best.pt"
        best_pth = self.stage3_dir / "best.pth"
        last_pt = self.stage3_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 3] Saved checkpoints: {best_pt} and {last_pt}")

        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage3"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 3 (RESOLUTION 1024)", metrics)
        return best_pt

    def run_stage4(
        self,
        stage3_weight: Path,
        epochs: int = 8,
        lr0: float = 0.0003,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
        erasing: float = 0.4,
        hsv_s: float = 0.8,
        hsv_v: float = 0.5,
    ) -> Path:
        """
        Stage 4: Force learning leaf characteristics via cutout/erasing and HSV jitter.
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 4 — AUGMENTATIONS FOR LEAF FEATURE LEARNING")
        print(f" Input model:     {stage3_weight}")
        print(f" Epochs:          {epochs}")
        print(f" Augmentations:   erasing={erasing}, hsv_s={hsv_s}, hsv_v={hsv_v}")
        print(f" Learning rate:   lr0={lr0}, lrf={lrf}")
        print(f" Output dir:      {self.stage4_dir}")
        print("=" * 78)

        self.stage4_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(stage3_weight), task="detect")

        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            erasing=erasing,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            device=self.device,
            workers=self.workers,
            project=str(self.stage4_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage4_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage4_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage4_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"best.pt weights not found in: {self.stage4_dir}")

        best_pt = self.stage4_dir / "best.pt"
        best_pth = self.stage4_dir / "best.pth"
        last_pt = self.stage4_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 4] Saved checkpoints: {best_pt} and {last_pt}")

        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage4"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 4 (AUGMENTATION)", metrics)
        return best_pt

    def run_stage5(
        self,
        stage4_weight: Path,
        epochs: int = 8,
        lr0: float = 0.0003,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 5: Train on images with yellow flowers desaturated (forcing leaf shape recognition).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 5 — TRAINING ON NO-YELLOW DATASET")
        print(f" Input model:     {stage4_weight}")
        print(f" Epochs:          {epochs}")
        print(f" Learning rate:   lr0={lr0}, lrf={lrf}")
        print(f" Output dir:      {self.stage5_dir}")
        print("=" * 78)

        self.stage5_dir.mkdir(parents=True, exist_ok=True)

        source_train_manifest = DATASETS_DIR / "stage2_train.txt"
        dataset_out = self.stage5_dir / "dataset"
        no_yellow_yaml = prepare_no_yellow_dataset(
            source_manifest=source_train_manifest,
            output_dataset_dir=dataset_out,
            val_manifest=self.val_manifest,
        )

        model = YOLO(str(stage4_weight), task="detect")

        model.train(
            data=str(no_yellow_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage5_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage5_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage5_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage5_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"best.pt weights not found in: {self.stage5_dir}")

        best_pt = self.stage5_dir / "best.pt"
        best_pth = self.stage5_dir / "best.pth"
        last_pt = self.stage5_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 5] Saved final model: {best_pt} and {last_pt}")

        # Evaluate on standard validation set (verify wild ragwort recognition)
        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage5"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 5 (NO-YELLOW FLOWERS)", metrics)
        return best_pt

    def save_comparison_summary(self):
        """Save final comparison summary across Stages 2, 3, 4, and 5."""
        summary_dir = self.output_dir / "leaves_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        json_path = summary_dir / "metrics_comparison.json"
        csv_path = summary_dir / "metrics_comparison.csv"
        txt_path = summary_dir / "summary.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)

        csv_lines = ["stage,checkpoint,mAP50,mAP50-95,precision,recall,fitness"]
        stages = ["stage2", "stage3", "stage4", "stage5"]
        for st in stages:
            if st in self.metrics_summary:
                info = self.metrics_summary[st]
                m = info["metrics"]
                csv_lines.append(
                    f"{st},{info['checkpoint']},{m.get('mAP50', 0):.4f},"
                    f"{m.get('mAP50-95', 0):.4f},{m.get('precision', 0):.4f},"
                    f"{m.get('recall', 0):.4f},{m.get('fitness', 0):.4f}"
                )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        header = "=" * 86 + "\n"
        title = "       LEAF STAGES SUMMARY: STAGE 2 -> STAGE 3 -> STAGE 4 -> STAGE 5\n"
        sep = "-" * 86 + "\n"
        col_names = f"{'Stage':<18} | {'mAP@50':<10} | {'mAP@50-95':<12} | {'Precision':<10} | {'Recall':<10} | {'Fitness':<10}\n"

        labels_map = {
            "stage2": "Stage 2 (Base)",
            "stage3": "Stage 3 (1024)",
            "stage4": "Stage 4 (Aug)",
            "stage5": "Stage 5 (NoYel)",
        }

        rows = []
        for st in stages:
            if st in self.metrics_summary:
                m = self.metrics_summary[st]["metrics"]
                lbl = labels_map.get(st, st)
                row = (
                    f"{lbl:<18} | "
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
        print(f"[Summary] Saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
        print(f"  - {txt_path}")

    def _print_metrics_block(self, title: str, metrics: Dict[str, float]):
        print(f"\n--- VALIDATION RESULTS: {title} ---")
        print(f"  mAP@50               : {metrics.get('mAP50', 0.0):.4f}")
        print(f"  mAP@50-95            : {metrics.get('mAP50-95', 0.0):.4f}")
        print(f"  Precision            : {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall               : {metrics.get('recall', 0.0):.4f}")
        print(f"  Fitness              : {metrics.get('fitness', 0.0):.4f}")
        print("---------------------------------------")


def run_leaves_fine_tuning(
    stage2_model_path: Optional[Union[str, Path]] = None,
    data_yaml: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    stage3_epochs: int = 8,
    stage4_epochs: int = 8,
    stage5_epochs: int = 8,
    lr0_stage3: float = 0.0005,
    lr0_stage4: float = 0.0003,
    lr0_stage5: float = 0.0003,
    imgsz: int = 1024,
    batch: int = 8,
    device: Optional[str] = None,
    workers: int = 4,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
    stage3_checkpoint: Optional[Union[str, Path]] = None,
    stage4_checkpoint: Optional[Union[str, Path]] = None,
    val_only: bool = False,
):
    """Main execution function for the leaf fine-tuning pipeline (Stages 3 -> 4 -> 5)."""
    resolved_model = resolve_model_path(stage2_model_path or STAGE2_DEFAULT_MODEL)
    inspect_model_architecture(resolved_model)

    pipeline = LeavesFineTuningPipeline(
        stage2_model_path=resolved_model,
        data_yaml=data_yaml,
        output_dir=output_dir,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
    )

    pipeline.evaluate_initial_stage2()
    if val_only:
        print("[Info] Ran in --val-only mode. Completed.")
        return pipeline.metrics_summary

    if skip_stage3:
        best_s3 = Path(stage3_checkpoint or (pipeline.stage3_dir / "best.pt")).resolve()
        print(f"\n[Info] Skipping Stage 3 training. Using existing checkpoint: {best_s3}")
        m = evaluate_checkpoint(best_s3, pipeline.data_yaml, device=pipeline.device, imgsz=imgsz, batch=batch)
        pipeline.metrics_summary["stage3"] = {"checkpoint": str(best_s3), "metrics": m}
    else:
        best_s3 = pipeline.run_stage3(
            epochs=stage3_epochs,
            lr0=lr0_stage3,
        )

    if skip_stage4:
        best_s4 = Path(stage4_checkpoint or (pipeline.stage4_dir / "best.pt")).resolve()
        print(f"\n[Info] Skipping Stage 4 training. Using existing checkpoint: {best_s4}")
        m = evaluate_checkpoint(best_s4, pipeline.data_yaml, device=pipeline.device, imgsz=imgsz, batch=batch)
        pipeline.metrics_summary["stage4"] = {"checkpoint": str(best_s4), "metrics": m}
    else:
        best_s4 = pipeline.run_stage4(
            stage3_weight=best_s3,
            epochs=stage4_epochs,
            lr0=lr0_stage4,
        )

    pipeline.run_stage5(
        stage4_weight=best_s4,
        epochs=stage5_epochs,
        lr0=lr0_stage5,
    )

    pipeline.save_comparison_summary()
    return pipeline.metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ragwort fine-tuning focused on leaf and structural features (Stage 3 -> 4 -> 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(STAGE2_DEFAULT_MODEL),
        help="Path to best Stage 2 checkpoint",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DATASETS_DIR / "stage2_data.yaml",
        help="Dataset YAML configuration from previous stage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINE_RESULTS_DIR,
        help="Main directory to store fine-tuning results",
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=8,
        help="Number of epochs for Stage 3 (1024 resolution)",
    )
    parser.add_argument(
        "--stage4-epochs",
        type=int,
        default=8,
        help="Number of epochs for Stage 4 (leaf augmentations)",
    )
    parser.add_argument(
        "--stage5-epochs",
        type=int,
        default=8,
        help="Number of epochs for Stage 5 (no-yellow flower images)",
    )
    parser.add_argument(
        "--lr0-stage3",
        type=float,
        default=0.0005,
        help="Learning rate for Stage 3",
    )
    parser.add_argument(
        "--lr0-stage4",
        type=float,
        default=0.0003,
        help="Learning rate for Stage 4",
    )
    parser.add_argument(
        "--lr0-stage5",
        type=float,
        default=0.0003,
        help="Learning rate for Stage 5",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image resolution for stages 3, 4, and 5",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (safe for 6GB VRAM at 1024 resolution)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Skip Stage 3 and proceed to Stage 4",
    )
    parser.add_argument(
        "--skip-stage4",
        action="store_true",
        help="Skip Stage 4 and proceed to Stage 5",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 3 checkpoint when skipped",
    )
    parser.add_argument(
        "--stage4-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 4 checkpoint when skipped",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Run validation only without training",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_leaves_fine_tuning(
        stage2_model_path=args.model,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        stage3_epochs=args.stage3_epochs,
        stage4_epochs=args.stage4_epochs,
        stage5_epochs=args.stage5_epochs,
        lr0_stage3=args.lr0_stage3,
        lr0_stage4=args.lr0_stage4,
        lr0_stage5=args.lr0_stage5,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        skip_stage3=args.skip_stage3,
        skip_stage4=args.skip_stage4,
        stage3_checkpoint=args.stage3_checkpoint,
        stage4_checkpoint=args.stage4_checkpoint,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()
