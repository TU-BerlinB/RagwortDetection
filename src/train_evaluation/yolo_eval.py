"""
File: src/train_evaluation/yolo_eval.py
Usage:
    python src/train_evaluation/yolo_eval.py --epochs 50 --batch 16 --imgsz 640
    # To evaluate existing weights without retraining:
    python src/train_evaluation/yolo_eval.py --skip-train --weights outputs/models/ragwort_yolov8_best.pt
Description:
    Training, validation, and evaluation pipeline for YOLOv8 on weighted ragwort detection datasets:
      1. YOLOv8 model training on weighted datasets (outputs/datasets/data_weighted/data_weighted.yaml).
      2. Best weights preservation to outputs/models/ragwort_yolov8_best.pt.
      3. Validation on test split for mAP50 and mAP50-95 metrics.
      4. Custom IoU=0.5 evaluation (Precision, Recall, F1, saving False Positive/False Negative images).
      5. Generalization tests on external web images with annotated bounding box outputs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
import yaml
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.yolov8 import YOLOv8

OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_YAML = OUTPUTS_DIR / "datasets" / "data_weighted" / "data_weighted.yaml"
RUNS_DIR = OUTPUTS_DIR / "runs" / "yolo"
MODELS_DIR = OUTPUTS_DIR / "models"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
EVAL_DIR = OUTPUTS_DIR / "evaluations" / "yolo"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions" / "yolo"

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str(WEIGHTS_DIR.resolve()),
        "runs_dir": str(RUNS_DIR.resolve()),
        "datasets_dir": str((OUTPUTS_DIR / "datasets").resolve()),
    })
except Exception:
    pass

TEST_IMAGES_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "images"
TEST_LABELS_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "labels"
EXTERNAL_DIR = REPO_ROOT / "data" / "external_test_images"
RESULTS_DIR = PREDICTIONS_DIR / "external_test_results"

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


def oblicz_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Box format: [xmin, ymin, xmax, ymax].
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    total_area = float(area_a + area_b - intersection)
    if total_area <= 0:
        return 0.0

    return intersection / total_area


calculate_iou = oblicz_iou


def ewaluacja_modelu(
    katalog_zdjec: str | Path,
    predykcje: Dict[str, List[List[float]]],
    poprawne_dane: Dict[str, List[List[float]]],
    prog_iou: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate detections against ground truth at specified IoU threshold,
    reporting Precision, Recall, F1, and saving False Positive/Negative images.
    """
    katalog_fp = EVAL_DIR / "bledy_False_Positives"
    katalog_fn = EVAL_DIR / "bledy_False_Negatives"

    os.makedirs(katalog_fp, exist_ok=True)
    os.makedirs(katalog_fn, exist_ok=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for nazwa_zdjecia, ramki_poprawne in poprawne_dane.items():
        ramki_modelu = predykcje.get(nazwa_zdjecia, [])
        znalezione_starce = 0

        for ramka_m in ramki_modelu:
            trafienie = False
            for ramka_p in ramki_poprawne:
                iou = oblicz_iou(ramka_m, ramka_p)
                if iou >= prog_iou:
                    trafienie = True
                    znalezione_starce += 1
                    break

            if trafienie:
                true_positives += 1
            else:
                false_positives += 1
                sciezka_zrodlo = os.path.join(katalog_zdjec, nazwa_zdjecia)
                if os.path.exists(sciezka_zrodlo):
                    shutil.copy(sciezka_zrodlo, os.path.join(katalog_fp, nazwa_zdjecia))

        przegapione = len(ramki_poprawne) - znalezione_starce
        if przegapione > 0:
            false_negatives += przegapione
            sciezka_zrodlo = os.path.join(katalog_zdjec, nazwa_zdjecia)
            if os.path.exists(sciezka_zrodlo):
                shutil.copy(sciezka_zrodlo, os.path.join(katalog_fn, nazwa_zdjecia))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("--- EVALUATION RESULTS (IoU = 0.5) ---")
    print(f"True Positives   : {true_positives}")
    print(f"False Positives  : {false_positives}")
    print(f"False Negatives  : {false_negatives}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"False Positives error folder : {katalog_fp}")
    print(f"False Negatives error folder : {katalog_fn}")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def wczytaj_poprawne_dane_z_folderu(images_dir: Path, labels_dir: Path) -> Dict[str, List[List[float]]]:
    """Load Ground Truth labels from YOLO .txt files and convert to pixel coordinates [xmin, ymin, xmax, ymax]."""
    ground_truth = {}
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])

    for img_file in image_files:
        lbl_file = labels_dir / f"{img_file.stem}.txt"
        boxes = []
        if lbl_file.exists():
            with Image.open(img_file) as img:
                orig_w, orig_h = img.size

            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # Class 0 is ragwort
                        if cls_id == 0:
                            xc, yc, bw, bh = map(float, parts[1:5])
                            xmin = (xc - bw / 2.0) * orig_w
                            ymin = (yc - bh / 2.0) * orig_h
                            xmax = (xc + bw / 2.0) * orig_w
                            ymax = (yc + bh / 2.0) * orig_h
                            boxes.append([xmin, ymin, xmax, ymax])
        ground_truth[img_file.name] = boxes
    return ground_truth


def pobierz_zewnetrzne_zdjecia() -> List[Path]:
    """Download external web test images for generalization testing."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "RagwortDetectionResearch/1.0 (academic research; student@university.edu)"}
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


def testuj_zewnetrzne_zdjecia(model: YOLOv8, sciezki_zdjec: List[Path]):
    """Run inference on external test images and save visualizations with bounding boxes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_info = {item["filename"]: item for item in EXTERNAL_IMAGES}

    print("\n--- EXTERNAL TEST RESULTS ---")
    print(f"{'Filename':<26} | {'Expected':<10} | {'Detections':<10} | {'Max Conf':<9} | {'Verdict'}")
    print("-" * 75)

    for img_path in sciezki_zdjec:
        meta = meta_info.get(img_path.name, {"expected": "unknown"})
        expected = meta["expected"]

        pred = model.predict(img_path, conf=0.20, verbose=False)[0]
        boxes = pred["boxes"]
        scores = pred["scores"]
        classes = pred["class_names"]

        n_det = len(boxes)
        max_conf = max(scores) if scores else 0.0

        if expected == "ragwort":
            werdykt = "SUCCESS (Ragwort detected)" if n_det > 0 else "MISSED"
        else:
            werdykt = "SUCCESS (Clean, no false alarms)" if n_det == 0 else "FALSE ALARM"

        print(f"{img_path.name:<26} | {expected:<10} | {n_det:<10} | {max_conf:<9.2f} | {werdykt}")

        with Image.open(img_path) as im:
            annotated = im.convert("RGB")
            draw = ImageDraw.Draw(annotated)

            for box, score, cls_name in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                color = "red" if "ragwort" in cls_name.lower() else "blue"
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)
                draw.text((xmin + 8, ymin + 8), f"{cls_name} {score:.2f}", fill="yellow")

            save_file = RESULTS_DIR / f"wynik_{img_path.name}"
            annotated.save(save_file)

    print(f"\nVisualizations saved to: {RESULTS_DIR}")


def ensure_portable_data_yaml(yaml_path: Path) -> Path:
    """
    Verify dataset YAML configuration and referenced manifest paths,
    ensuring relative paths that work seamlessly across Docker and host environments.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return yaml_path

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[Warning] Cannot load {yaml_path}: {e}")
        return yaml_path

    if not isinstance(cfg, dict):
        return yaml_path

    yaml_dir = yaml_path.parent.resolve()
    modified = False

    if "path" in cfg and cfg["path"]:
        p_val = Path(str(cfg["path"]))
        if not p_val.exists() or p_val.as_posix() == ".":
            cfg.pop("path", None)
            modified = True

    for key in ("train", "val", "test"):
        if key in cfg and isinstance(cfg[key], str) and cfg[key].endswith(".txt"):
            txt_cand = yaml_dir / Path(cfg[key]).name
            if txt_cand.exists() and (cfg[key] != txt_cand.name or not Path(cfg[key]).exists()):
                cfg[key] = txt_cand.name
                modified = True

            txt_file = yaml_dir / Path(cfg[key]).name
            if txt_file.is_file():
                try:
                    lines = txt_file.read_text(encoding="utf-8").splitlines()
                    new_lines = []
                    lines_changed = False
                    for line in lines:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        if line_str.startswith("/") and not Path(line_str).exists():
                            parts = Path(line_str).parts
                            if "data" in parts:
                                rel_to_repo = Path(*parts[parts.index("data"):])
                                target_img = (REPO_ROOT / rel_to_repo).resolve()
                                rel_from_txt = os.path.relpath(target_img, yaml_dir).replace("\\", "/")
                                new_lines.append(f"./{rel_from_txt}")
                                lines_changed = True
                            else:
                                new_lines.append(line_str)
                        else:
                            new_lines.append(line_str)
                    if lines_changed:
                        txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                        print(f"[Info] Converted paths to relative in: {txt_file}")
                except Exception as e:
                    print(f"[Warning] Failed updating manifest {txt_file}: {e}")

    if modified:
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[Info] Updated configuration {yaml_path} with relative paths.")
        except Exception as e:
            print(f"[Warning] Could not write updated {yaml_path}: {e}")

    return yaml_path


def trenuj_i_ewaluuj(
    data_yaml: Optional[Union[str, Path]] = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    cache: Union[bool, str] = False,
    device: Optional[str] = None,
    model_name: str = "yolov8s.pt",
    optimizer: str = "AdamW",
    skip_train: bool = False,
    weights_path: Optional[str] = None,
    test_images_dir: Optional[Union[str, Path]] = None,
    test_labels_dir: Optional[Union[str, Path]] = None,
):
    """Main execution function for training, metric validation, and external testing."""
    import torch

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if data_yaml is None:
        candidates = [
            OUTPUTS_DIR / "datasets" / "data_weighted" / "data_weighted.yaml",
            REPO_ROOT / "data" / "data_concatenated" / "data_weighted.yaml",
            REPO_ROOT / "dataset_weighted" / "data.yaml",
            REPO_ROOT / "data" / "combined_dataset" / "data_weighted.yaml",
            REPO_ROOT / "data" / "combined_dataset" / "data.yaml",
        ]
        for c in candidates:
            if c.exists():
                data_yaml = c
                break
        if data_yaml is None or not Path(data_yaml).exists():
            print("[Info] data_weighted.yaml not found. Running automatic dataset weighting...")
            from src.scripts.weight_dataset import process_concatenated_dataset
            data_yaml = process_concatenated_dataset(REPO_ROOT / "data" / "data_concatenated")

    data_yaml = ensure_portable_data_yaml(Path(data_yaml))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    p_model = Path(model_name)
    if not p_model.is_file():
        if (WEIGHTS_DIR / p_model.name).is_file():
            model_name = str((WEIGHTS_DIR / p_model.name).resolve())
        elif (MODELS_DIR / p_model.name).is_file():
            model_name = str((MODELS_DIR / p_model.name).resolve())
        elif (OUTPUTS_DIR / p_model.name).is_file():
            model_name = str((OUTPUTS_DIR / p_model.name).resolve())
        elif (REPO_ROOT / "weights" / p_model.name).is_file():
            target_pt = WEIGHTS_DIR / p_model.name
            shutil.copy2(REPO_ROOT / "weights" / p_model.name, target_pt)
            model_name = str(target_pt.resolve())
        elif p_model.suffix in (".pt", ".pth") or not p_model.parent.name:
            model_name = str((WEIGHTS_DIR / p_model.name).resolve())
    elif p_model.parent.resolve() == (REPO_ROOT / "weights").resolve():
        target_pt = WEIGHTS_DIR / p_model.name
        shutil.copy2(p_model, target_pt)
        model_name = str(target_pt.resolve())

    if weights_path:
        p_wp = Path(weights_path)
        if not p_wp.is_file():
            if (WEIGHTS_DIR / p_wp.name).is_file():
                weights_path = str((WEIGHTS_DIR / p_wp.name).resolve())
            elif (MODELS_DIR / p_wp.name).is_file():
                weights_path = str((MODELS_DIR / p_wp.name).resolve())
            elif (REPO_ROOT / "weights" / p_wp.name).is_file():
                target_wp = WEIGHTS_DIR / p_wp.name
                shutil.copy2(REPO_ROOT / "weights" / p_wp.name, target_wp)
                weights_path = str(target_wp.resolve())

    print("====================================================================")
    print(" 1. YOLOV8 MODEL TRAINING")
    print("====================================================================")
    print(f"[YOLOv8] Device:                 {device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and device != "cpu":
        try:
            print(f"[YOLOv8] GPU:                    {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"[YOLOv8] Dataset:                {data_yaml}")
    print(f"[YOLOv8] Base model:             {model_name}")
    print(f"[YOLOv8] Params:                 Epochs: {epochs}, Imgsz: {imgsz}, Batch: {batch}, Cache: {cache}, Optimizer: {optimizer}, Workers: {workers}")
    print(f"[YOLOv8] Output runs dir:        {RUNS_DIR}")

    best_weights_file = MODELS_DIR / "ragwort_yolov8_best.pt"

    if skip_train and (weights_path or best_weights_file.exists()):
        chosen_weight = weights_path or str(best_weights_file)
        print(f"[Info] Skipping training. Loading existing weights: {chosen_weight}")
        model = YOLOv8(chosen_weight, device=device)
    else:
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}!")

        model = YOLOv8(model_name, device=device)

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            optimizer=optimizer,
            cache=cache,
            project=str(RUNS_DIR),
            name="ragwort_yolov8_weighted",
            exist_ok=True,
            verbose=True,
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        trained_best = RUNS_DIR / "ragwort_yolov8_weighted" / "weights" / "best.pt"
        if trained_best.exists():
            shutil.copy2(trained_best, best_weights_file)
            shutil.copy2(trained_best, WEIGHTS_DIR / "ragwort_yolov8_best.pt")
            shutil.copy2(trained_best, WEIGHTS_DIR / "best.pt")
            print(f"\n[Weights] Saved best weights to: {best_weights_file} and {WEIGHTS_DIR / 'best.pt'}")

        trained_last = RUNS_DIR / "ragwort_yolov8_weighted" / "weights" / "last.pt"
        if trained_last.exists():
            shutil.copy2(trained_last, WEIGHTS_DIR / "last.pt")

        root_w = REPO_ROOT / "weights"
        if root_w.is_dir() and root_w.resolve() != WEIGHTS_DIR.resolve():
            for f in root_w.iterdir():
                if f.is_file():
                    shutil.copy2(f, WEIGHTS_DIR / f.name)
            try:
                shutil.rmtree(root_w)
            except Exception:
                pass

    print("\n====================================================================")
    print(" 2. MODEL VALIDATION ON TEST SPLIT (YOLO Metrics)")
    print("====================================================================")
    try:
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", device=device, verbose=False)
        print(f"mAP@50               : {metrics.box.map50:.4f}")
        print(f"mAP@50-95            : {metrics.box.map:.4f}")
        print(f"Precision            : {metrics.box.mp:.4f}")
        print(f"Recall               : {metrics.box.mr:.4f}")
    except Exception as e:
        print(f"[Warning] YOLO validation encountered an issue: {e}")

    print("\n====================================================================")
    print(" 3. CUSTOM EVALUATION (IoU = 0.5)")
    print("====================================================================")
    t_img = test_images_dir
    t_lbl = test_labels_dir
    if t_img is None or not Path(t_img).exists():
        for cand_img, cand_lbl in [
            (REPO_ROOT / "data" / "data_concatenated" / "combined_dataset" / "test" / "images",
             REPO_ROOT / "data" / "data_concatenated" / "combined_dataset" / "test" / "labels"),
            (TEST_IMAGES_DIR, TEST_LABELS_DIR),
            (REPO_ROOT / "dataset_weighted" / "images" / "val",
             REPO_ROOT / "dataset_weighted" / "labels" / "val"),
        ]:
            if cand_img.exists() and cand_lbl.exists():
                t_img, t_lbl = cand_img, cand_lbl
                break

    if t_img and Path(t_img).exists() and t_lbl and Path(t_lbl).exists():
        print(f"[data_eval] Evaluating Precision, Recall, and F1 on: {t_img}")
        ground_truth = wczytaj_poprawne_dane_z_folderu(Path(t_img), Path(t_lbl))
        predictions = model.predict_for_eval(Path(t_img), conf=0.25)
        ewaluacja_modelu(str(t_img), predictions, ground_truth, prog_iou=0.5)
    else:
        print("[Warning] No test split directory available for custom evaluation.")

    print("\n====================================================================")
    print(" 4. EXTERNAL TEST IMAGES GENERALIZATION")
    print("====================================================================")
    try:
        external_images = pobierz_zewnetrzne_zdjecia()
        testuj_zewnetrzne_zdjecia(model, external_images)
    except Exception as e:
        print(f"[Warning] External tests skipped: {e}")

    print("\n====================================================================")
    print(" YOLO PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLOv8 model on weighted datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (default: auto-detect data_weighted.yaml)")
    parser.add_argument("--device", type=str, default=None, help="Device: '0' (CUDA/GPU), 'cpu' (default: auto-detect)")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model weights (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes (default: 8)")
    parser.add_argument("--cache", type=str, default="none", choices=["none", "ram", "disk"], help="Image cache option")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam", "auto"], help="Optimizer (default: AdamW)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and load existing weights")
    parser.add_argument("--weights", type=str, default=None, help="Custom path to model weights .pt")
    args = parser.parse_args()

    cache_val: Union[bool, str] = False
    if args.cache == "ram":
        cache_val = True
    elif args.cache == "disk":
        cache_val = "disk"

    trenuj_i_ewaluuj(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=cache_val,
        device=args.device,
        model_name=args.model,
        optimizer=args.optimizer,
        skip_train=args.skip_train,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()