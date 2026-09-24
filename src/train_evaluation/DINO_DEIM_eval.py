"""
File: src/train_evaluation/DINO_DEIM_eval.py
Usage:
    python src/train_evaluation/DINO_DEIM_eval.py --epochs 30 --batch-size 4 --device cuda
    python src/train_evaluation/DINO_DEIM_eval.py --smoke-test --batch-size 4
    python src/train_evaluation/DINO_DEIM_eval.py --skip-train --resume outputs/checkpoints/deimv2_dinov3_ragwort/best_stg1.pth
Description:
    Training, validation, and evaluation pipeline for DEIMv2 + DINOv3 on weighted datasets:
      1. Automatic weighted COCO dataset generation without image duplication.
      2. DEIMv2 detector training (DINOv3 backbone) optimized for GPU VRAM.
      3. Standard COCO evaluation (mAP50, mAP50-95, APs, APm, APl).
      4. Custom IoU=0.5 evaluation (Precision, Recall, F1, error analysis saving FP/FN).
      5. Generalization inference on external images with bounding box visualizations.
      6. Best checkpoint export to model.pt and ragwort_deimv2_best.pth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_deimv2_root() -> Path:
    """Detect DEIMv2 repository directory in Docker container or on the host."""
    candidates = [
        Path("/DEIMv2"),
        PROJECT_ROOT.parent / "DEIMv2",
        PROJECT_ROOT.parent.parent / "DEIMv2",
    ]
    for c in candidates:
        if c.is_dir() and (c / "train.py").is_file():
            return c.resolve()
    return (PROJECT_ROOT.parent / "DEIMv2").resolve()


DEIMV2_ROOT = find_deimv2_root()
if str(DEIMV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DEIMV2_ROOT))

DEFAULT_CONFIG = DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_dinov3_ragwort.yml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs" / "deim" / "ragwort_deimv2_weighted"
MODELS_DIR = OUTPUTS_DIR / "models"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
EVAL_DIR = OUTPUTS_DIR / "evaluations" / "deim"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions" / "deim"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_test_images"
RESULTS_DIR = PREDICTIONS_DIR / "external_test_results"
DATA_CONCAT_DIR = PROJECT_ROOT / "data" / "data_concatenated"
DATASETS_COCO_DIR = OUTPUTS_DIR / "datasets" / "coco"
LEGACY_COCO_DIR = OUTPUTS_DIR / "combined_dataset_coco"


def find_data_concat_dir() -> Path:
    """Find directory containing input datasets (Felix_data, combined_dataset, etc.)."""
    candidates = [
        DATA_CONCAT_DIR,
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "RagwortYOLO" / "RagwortDetection" / "data" / "data_concatenated",
    ]
    for c in candidates:
        if c.is_dir() and ((c / "Felix_data").is_dir() or (c / "combined_dataset").is_dir()):
            return c.resolve()
    return DATA_CONCAT_DIR


EXTERNAL_IMAGES = [
    {
        "filename": "ragwort_field_1.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Jacobaea_vulgaris-3235.jpg",
        "expected": "ragwort",
        "description": "Ragwort (Jacobaea vulgaris) blooming in a meadow",
    },
    {
        "filename": "ragwort_flowers_2.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/%28MHNT%29_Halictus_rubicundus_on_Jacobaea_vulgaris_-_Villeneuve-les-Bouloc_France.jpg",
        "expected": "ragwort",
        "description": "Close-up of ragwort flowers with an insect",
    },
    {
        "filename": "negative_dandelion_3.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b5/Gesloten_bloem_van_de_paardenbloem_%28Taraxacum_officinale%29_09-05-2021._%28d.j.b%29_02.jpg",
        "expected": "negative",
        "description": "Dandelion flower (negative sample with different yellow flowers)",
    },
    {
        "filename": "negative_meadow_4.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cd/Sch%C3%B6nwald_im_Schwarzwald%2C_Escheckstra%C3%9Fe_--_2025_--_0150.jpg",
        "expected": "negative",
        "description": "Green meadow and grass without ragwort (negative sample)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to DEIMv2 YAML config")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4 for 6GB VRAM)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for detector (default: 0.0005)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--good-weight", type=int, default=15, help="Oversampling multiplier for Felix_data (default: 15x)")
    parser.add_argument("--other-weight", type=int, default=1, help="Multiplier for other data (default: 1x)")
    parser.add_argument("--train-backbone", action="store_true", help="Unfreeze and train DINOv3 ViT backbone")
    parser.add_argument("--resume", type=Path, default=None, help="Path to .pth checkpoint to resume or evaluate")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluation directly")
    parser.add_argument("--evaluate", action="store_true", help="Run COCO evaluation only (--test-only)")
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test on a single train and validation batch")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Force rebuilding weighted COCO dataset")
    parser.add_argument("--export-model", action="store_true", help="Export existing checkpoint to model.pt")
    return parser.parse_args()


def ensure_weighted_coco_dataset(good_weight: int = 15, other_weight: int = 1, force: bool = False) -> Dict[str, Path]:
    """Ensure weighted COCO dataset exists and contains samples, building it if necessary."""
    train_ann = DATASETS_COCO_DIR / "train" / "annotations.json"
    val_ann = DATASETS_COCO_DIR / "val" / "annotations.json"

    if not force and train_ann.is_file() and val_ann.is_file():
        try:
            with open(train_ann, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data.get("images", [])) > 0:
                return {
                    "train": train_ann,
                    "val": val_ann,
                    "test": DATASETS_COCO_DIR / "test" / "annotations.json",
                    "img_folder": DATASETS_COCO_DIR,
                }
        except Exception:
            pass

    concat_dir = find_data_concat_dir()
    print("\n[DINO_DEIM] Weighted COCO dataset not found or rebuild requested.")
    print(f"[DINO_DEIM] Running dataset weighting on: {concat_dir}")

    from src.scripts.weight_dataset import process_concatenated_dataset

    process_concatenated_dataset(
        concatenated_dir=concat_dir,
        good_weight=good_weight,
        other_weight=other_weight,
        export_coco=True,
    )

    return {
        "train": train_ann,
        "val": val_ann,
        "test": DATASETS_COCO_DIR / "test" / "annotations.json",
        "img_folder": DATASETS_COCO_DIR,
    }


def update_items(args: argparse.Namespace) -> list[str]:
    """Generate list of configuration overrides for DEIMv2 CLI."""
    coco_root = DATASETS_COCO_DIR if (DATASETS_COCO_DIR / "train" / "annotations.json").is_file() else LEGACY_COCO_DIR
    output_dir = RUNS_DIR

    items: list[str] = [
        f"output_dir={output_dir}",
        f"train_dataloader.dataset.img_folder={coco_root / 'train' / 'images'}",
        f"train_dataloader.dataset.ann_file={coco_root / 'train' / 'annotations.json'}",
        f"val_dataloader.dataset.img_folder={coco_root / 'val' / 'images'}",
        f"val_dataloader.dataset.ann_file={coco_root / 'val' / 'annotations.json'}",
        f"test_dataloader.dataset.img_folder={coco_root / 'test' / 'images'}",
        f"test_dataloader.dataset.ann_file={coco_root / 'test' / 'annotations.json'}",
    ]

    if args.epochs is not None:
        epochs = args.epochs
        stop_epoch = max(2, int(epochs * 0.90))
        flat_epoch = max(1, int(epochs * 0.50))
        no_aug_epoch = max(1, int(epochs * 0.10))
        mixup_start = max(1, int(epochs * 0.05))
        mixup_end = max(2, int(epochs * 0.50))
        copyblend_start = max(1, int(epochs * 0.05))

        items.extend([
            f"epoches={epochs}",
            f"flat_epoch={flat_epoch}",
            f"no_aug_epoch={no_aug_epoch}",
            f"train_dataloader.collate_fn.stop_epoch={stop_epoch}",
            f"train_dataloader.collate_fn.mixup_epochs=[{mixup_start},{mixup_end}]",
            f"train_dataloader.collate_fn.copyblend_epochs=[{copyblend_start},{stop_epoch}]",
        ])

    if args.batch_size is not None:
        items.extend([
            f"train_dataloader.total_batch_size={args.batch_size}",
            f"val_dataloader.total_batch_size={args.batch_size}",
        ])
    if args.lr is not None:
        items.append(f"optimizer.lr={args.lr}")
    if args.workers is not None:
        items.extend([
            f"train_dataloader.num_workers={args.workers}",
            f"val_dataloader.num_workers={args.workers}",
        ])

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    items.append(f"device={device_str}")

    if args.resume is not None:
        items.append(f"resume={Path(args.resume).resolve()}")
    if args.train_backbone:
        items.append("DINOv3STAs.finetune=True")

    return items


def oblicz_iou(box_a: List[float], box_b: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    total_area = float(area_a + area_b - intersection_area)
    return (intersection_area / total_area) if total_area > 0 else 0.0


calculate_iou = oblicz_iou


def load_deim_model(config_path: Path, checkpoint_path: Optional[Path], device: torch.device):
    """Load DEIMv2 + DINOv3 model architecture and postprocessor."""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt_p = Path(checkpoint_path)
        print(f"[DEIM] Loading checkpoint weights: {ckpt_p}")
        data = torch.load(ckpt_p, map_location="cpu")
        state_dict = (
            data.get("ema", {}).get("module")
            or data.get("model")
            or data.get("model_state_dict")
            or data
        )
        model.load_state_dict(state_dict, strict=False)
        print("[DEIM] Model weights loaded successfully!")
    return model, postprocessor


def predict_deim_single(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    conf: float = 0.25,
) -> Dict[str, Any]:
    """Execute DEIMv2 detection inference on a single PIL Image."""
    w, h = image.size
    orig_size = torch.tensor([[h, w]], device=device)

    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tensor_in = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_in)
        preds = postprocessor(outputs, orig_size)[0]

    scores = preds["scores"].cpu()
    boxes = preds["boxes"].cpu()
    labels = preds["labels"].cpu()

    keep = scores >= conf
    filtered_boxes = boxes[keep].tolist()
    filtered_scores = scores[keep].tolist()
    filtered_labels = labels[keep].tolist()

    return {
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "class_names": ["ragwort" for _ in filtered_labels],
    }


def ewaluacja_modelu_deim(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    images_dir: Path,
    labels_dir: Path,
    prog_iou: float = 0.5,
    conf_threshold: float = 0.25,
) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1 at specified IoU threshold, saving FP and FN error images."""
    fp_dir = EVAL_DIR / "bledy_False_Positives"
    fn_dir = EVAL_DIR / "bledy_False_Negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    for img_path in image_files:
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            preds = predict_deim_single(model, postprocessor, im, device, conf=conf_threshold)

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        ground_truth_boxes: List[List[float]] = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                p = line.strip().split()
                if len(p) >= 5 and int(p[0]) == 0:  # ragwort
                    xc, yc, bw, bh = map(float, p[1:5])
                    xmin = (xc - bw / 2.0) * orig_w
                    ymin = (yc - bh / 2.0) * orig_h
                    xmax = (xc + bw / 2.0) * orig_w
                    ymax = (yc + bh / 2.0) * orig_h
                    ground_truth_boxes.append([xmin, ymin, xmax, ymax])

        model_boxes = preds["boxes"]
        matched_gt_count = 0

        for m_box in model_boxes:
            hit = False
            for gt_box in ground_truth_boxes:
                if oblicz_iou(m_box, gt_box) >= prog_iou:
                    hit = True
                    matched_gt_count += 1
                    break

            if hit:
                true_positives += 1
            else:
                false_positives += 1
                try:
                    shutil.copy2(img_path, fp_dir / f"deim_{img_path.name}")
                except Exception:
                    pass

        missed = len(ground_truth_boxes) - matched_gt_count
        if missed > 0:
            false_negatives += missed
            try:
                shutil.copy2(img_path, fn_dir / f"deim_{img_path.name}")
            except Exception:
                pass

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n--- DEIMv2 DETECTION EVALUATION RESULTS (IoU = 0.5) ---")
    print(f"True Positives  : {true_positives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1-Score        : {f1:.4f}")
    print(f"False Positives error folder : {fp_dir}")
    print(f"False Negatives error folder : {fn_dir}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def pobierz_zewnetrzne_zdjecia() -> List[Path]:
    """Download external verification images from the web."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "RagwortDetectionDEIM/1.0 (academic research; student@university.edu)"}
    paths = []
    print("\n--- DOWNLOADING EXTERNAL TEST IMAGES ---")
    for item in EXTERNAL_IMAGES:
        dest = EXTERNAL_DIR / item["filename"]
        if not dest.exists():
            print(f"Downloading: {item['filename']} ({item['description']})...")
            try:
                r = requests.get(item["url"], headers=headers, timeout=20)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    print(f"  -> Success ({len(r.content) // 1024} KB)")
                else:
                    print(f"  -> HTTP Error: {r.status_code}")
            except Exception as e:
                print(f"  -> Exception: {e}")
        else:
            print(f"File exists locally: {item['filename']}")
        if dest.exists():
            paths.append(dest)
    return paths


def testuj_zewnetrzne_zdjecia_deim(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    image_paths: List[Path],
):
    """Run DEIM inference on external test images and save visualizations with bounding boxes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_info = {item["filename"]: item for item in EXTERNAL_IMAGES}

    print("\n--- DEIMv2 EXTERNAL IMAGE INFERENCE RESULTS ---")
    print(f"{'Filename':<26} | {'Expected':<10} | {'Detections':<10} | {'Max Conf':<9} | {'Verdict'}")
    print("-" * 75)

    for img_path in image_paths:
        meta = meta_info.get(img_path.name, {"expected": "unknown"})
        expected = meta["expected"]

        with Image.open(img_path) as im:
            preds = predict_deim_single(model, postprocessor, im, device, conf=0.20)
            annotated = im.convert("RGB")

        boxes = preds["boxes"]
        scores = preds["scores"]
        classes = preds["class_names"]

        n_det = len(boxes)
        max_conf = max(scores) if scores else 0.0

        if expected == "ragwort":
            verdict = "SUCCESS (Ragwort detected)" if n_det > 0 else "MISSED"
        else:
            verdict = "SUCCESS (Clean, no false alarms)" if n_det == 0 else "FALSE ALARM"

        print(f"{img_path.name:<26} | {expected:<10} | {n_det:<10} | {max_conf:<9.2f} | {verdict}")

        draw = ImageDraw.Draw(annotated)
        for box, score, cls_name in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            color = "red" if "ragwort" in cls_name.lower() else "blue"
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)
            draw.text((xmin + 8, ymin + 8), f"{cls_name} {score:.2f}", fill="yellow")

        save_file = RESULTS_DIR / f"deim_wynik_{img_path.name}"
        annotated.save(save_file)

    print(f"\nVisualizations saved to: {RESULTS_DIR}")


def smoke_test(config_path: Path, overrides: list[str]) -> None:
    """Fast validation of a single batch: forward pass, loss calculation, backward step, and evaluation."""
    from engine.core import YAMLConfig, yaml_utils
    from engine.solver.det_engine import evaluate

    cfg = YAMLConfig(str(config_path), **yaml_utils.parse_cli(overrides))
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, criterion = cfg.model.to(device), cfg.criterion.to(device)
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    print(f"Smoke batch: images={tuple(samples.shape)}; target boxes={[tuple(t['boxes'].shape) for t in targets]}")
    model.train()
    criterion.train()
    outputs = model(samples, targets=targets)
    loss_dict = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    loss = sum(loss_dict.values())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Invalid training loss: {loss_dict}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable_with_grad = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    frozen_with_grad = sum(p.grad is not None for p in model.backbone.dinov3.parameters() if not p.requires_grad)
    optimizer.step()
    print(f"Smoke train step: loss={loss.item():.4f}; trainable with grad={trainable_with_grad}; frozen DINOv3 grads={frozen_with_grad}")

    evaluator = cfg.evaluator
    one_val_batch = [next(iter(cfg.val_dataloader))]
    stats, _ = evaluate(model, criterion, cfg.postprocessor.to(device), one_val_batch, evaluator, device)
    bbox = stats.get("coco_eval_bbox", [])
    if bbox:
        print(f"Smoke validation: AP={bbox[0]:.4f}, AP50={bbox[1]:.4f}, AP75={bbox[2]:.4f}")


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best checkpoint to export. Priority: best_stg1.pth -> best.pth -> last.pth."""
    candidates = [
        output_dir / "best_stg1.pth",
        output_dir / "best.pth",
        output_dir / "last.pth",
    ]
    for checkpoint in candidates:
        if checkpoint.is_file():
            print(f"[Export] Found checkpoint: {checkpoint}")
            return checkpoint

    checkpoints = sorted(
        output_dir.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        print(f"[Export] Selected most recent checkpoint: {checkpoints[0]}")
        return checkpoints[0]

    raise FileNotFoundError(f"No .pth checkpoint found in {output_dir}")


def export_model_pt(config_path: Path, overrides: list[str]) -> Path:
    """Convert DEIMv2 checkpoint to model.pt format containing config and CPU weights."""
    from engine.core import YAMLConfig, yaml_utils

    cfg = YAMLConfig(str(config_path), **yaml_utils.parse_cli(overrides))
    raw = cfg.yaml_cfg
    output_dir = Path(raw.get("output_dir", RUNS_DIR)).resolve()
    checkpoint_path = find_best_checkpoint(output_dir)

    print("\n" + "=" * 70)
    print(" EXPORTING MODEL TO model.pt")
    print("=" * 70)
    print(f"Source checkpoint: {checkpoint_path}")

    model = cfg.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned_state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys during loading ({len(missing_keys)})")
    if unexpected_keys:
        print(f"Warning: unexpected keys during loading ({len(unexpected_keys)})")

    model.eval()
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ann_file = Path(raw.get("train_dataloader", {}).get("dataset", {}).get("ann_file", ""))
    category_data = []
    if ann_file.is_file():
        try:
            coco_meta = json.loads(ann_file.read_text(encoding="utf-8"))
            category_data = coco_meta.get("categories", [])
        except Exception:
            pass

    export_data = {
        "model_state_dict": state_dict_cpu,
        "checkpoint": str(checkpoint_path),
        "config": raw,
        "categories": category_data,
        "num_classes": raw.get("num_classes", 1),
        "format": "DEIMv2-DINOv3",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_pt = MODELS_DIR / "model.pt"
    torch.save(export_data, model_pt)

    run_model_pt = output_dir / "model.pt"
    try:
        shutil.copy2(model_pt, run_model_pt)
    except Exception:
        pass

    print(f"[Export] Successfully saved model: {model_pt}")
    print(f"[Export] File size: {model_pt.stat().st_size / (1024 ** 2):.2f} MB")
    print("=" * 70)
    return model_pt


def main() -> None:
    args = parse_args()

    ensure_weighted_coco_dataset(
        good_weight=args.good_weight,
        other_weight=args.other_weight,
        force=args.rebuild_dataset,
    )

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    os.chdir(DEIMV2_ROOT)
    overrides = update_items(args)

    if args.export_model:
        export_model_pt(config_path, overrides)
        return

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("=" * 75)
    print(" 1. DEIMv2 + DINOv3 CONFIGURATION")
    print("=" * 75)
    print(f"[DEIMv2] Device:                 {device}")
    if torch.cuda.is_available() and "cuda" in str(device):
        try:
            print(f"[DEIMv2] GPU:                    {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"[DEIMv2] DEIMv2 root:            {DEIMV2_ROOT}")
    print(f"[DEIMv2] Config:                 {config_path}")
    print(f"[DEIMv2] Hyperparams:            Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Workers: {args.workers}")
    print(f"[DEIMv2] Dataset weights:        Felix_data: {args.good_weight}x, Other: {args.other_weight}x")

    if args.smoke_test:
        print("\n--- RUNNING SINGLE BATCH SMOKE TEST ---")
        smoke_test(config_path, overrides)
        print("\n[SUCCESS] Smoke test completed successfully!")
        return

    output_ckpt_dir = RUNS_DIR
    best_weights_file = output_ckpt_dir / "best_stg1.pth"
    project_best_weights = MODELS_DIR / "ragwort_deimv2_best.pth"

    # Step 1: Training
    if not args.skip_train and not args.evaluate:
        print("\n====================================================================")
        print(" 2. STARTING DEIMv2 TRAINING")
        print("====================================================================")
        cmd = [sys.executable, "train.py", "--config", str(config_path), "--seed", "0"]
        if overrides:
            cmd.extend(("--update", *overrides))
        print(f"[DEIMv2] Command: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=DEIMV2_ROOT, check=True)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        chosen_ckpt = best_weights_file if best_weights_file.exists() else (output_ckpt_dir / "last.pth")
        if chosen_ckpt.exists():
            shutil.copy2(chosen_ckpt, project_best_weights)
            print(f"\n[Weights] Saved best weights to: {project_best_weights}")

        try:
            export_model_pt(config_path, overrides)
        except Exception as e:
            print(f"[Warning] Automatic export to model.pt failed: {e}")

    active_weights = args.resume or (project_best_weights if project_best_weights.exists() else best_weights_file)
    if not active_weights or not Path(active_weights).exists():
        cand_last = output_ckpt_dir / "last.pth"
        if cand_last.exists():
            active_weights = cand_last

    # Step 2: COCO validation
    print("\n====================================================================")
    print(" 3. COCO VALIDATION (mAP50, mAP50-95)")
    print("====================================================================")
    if active_weights and Path(active_weights).exists():
        val_cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
            "--test-only",
            "-r",
            str(Path(active_weights).resolve()),
        ]
        if overrides:
            val_cmd.extend(("--update", *overrides))
        try:
            subprocess.run(val_cmd, cwd=DEIMV2_ROOT, check=True)
        except Exception as e:
            print(f"[Warning] COCO validation encountered an issue: {e}")
    else:
        print("[Warning] No trained weights available for COCO validation.")

    # Step 3: Detection evaluation (IoU = 0.5)
    print("\n====================================================================")
    print(" 4. EVALUATION WITH CUSTOM DETECTION METRICS (IoU = 0.5)")
    print("====================================================================")
    concat_dir = find_data_concat_dir()
    test_img_dir = concat_dir / "combined_dataset" / "test" / "images"
    test_lbl_dir = concat_dir / "combined_dataset" / "test" / "labels"

    if not (test_img_dir.exists() and test_lbl_dir.exists()):
        alt_img_dir = PROJECT_ROOT / "data" / "combined_dataset" / "test" / "images"
        alt_lbl_dir = PROJECT_ROOT / "data" / "combined_dataset" / "test" / "labels"
        if alt_img_dir.exists() and alt_lbl_dir.exists():
            test_img_dir, test_lbl_dir = alt_img_dir, alt_lbl_dir

    if test_img_dir.exists() and test_lbl_dir.exists():
        try:
            model, postproc = load_deim_model(config_path, active_weights, device)
            ewaluacja_modelu_deim(
                model=model,
                postprocessor=postproc,
                device=device,
                images_dir=test_img_dir,
                labels_dir=test_lbl_dir,
                prog_iou=0.5,
                conf_threshold=0.25,
            )
        except Exception as e:
            print(f"[Warning] Detection evaluation encountered an error: {e}")
    else:
        print(f"[Warning] Test directory {test_img_dir} does not exist - skipping custom evaluation.")

    # Step 4: External image generalization test
    print("\n====================================================================")
    print(" 5. EXTERNAL IMAGES GENERALIZATION TEST")
    print("====================================================================")
    try:
        if 'model' not in locals():
            model, postproc = load_deim_model(config_path, active_weights, device)
        external_images = pobierz_zewnetrzne_zdjecia()
        testuj_zewnetrzne_zdjecia_deim(model, postproc, device, external_images)
    except Exception as e:
        print(f"[Warning] External tests skipped: {e}")

    print("\n====================================================================")
    print(" DINO/DEIM PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================================")


if __name__ == "__main__":
    main()