#!/usr/bin/env python3
"""
DINO_DEIM_eval.py - Trening, walidacja i ewaluacja modelu DEIMv2 + DINOv3 (ze zbalansowanym/zwagowanym zbiorem danych).

Funkcjonalności:
  1. Automatyczne generowanie zwagowanego zbioru COCO (Felix_data np. 15x, pozostałe 1x) bez kopiowania zdjęć.
  2. Trening detektora DEIMv2 (DINOv3 backbone) z optymalnym harmonogramem epok pod GPU RTX (np. 6 GB VRAM).
  3. Oficjalna ewaluacja COCO (mAP50, mAP50-95, APs, APm, APl).
  4. Ewaluacja data_eval (IoU = 0.5, Precision, Recall, F1, True Positives, zapis błędów FP/FN).
  5. Testy na zewnętrznych zdjęciach z internetu z rysowaniem ramek i werdyktem.
  6. Zapis najlepszego checkpointu do models/ragwort_deimv2_best.pth.

Przykłady użycia:
  python src/train_evaluation/DINO_DEIM_eval.py --epochs 30 --batch-size 4 --device cuda
  python src/train_evaluation/DINO_DEIM_eval.py --smoke-test --batch-size 4
  python src/train_evaluation/DINO_DEIM_eval.py --skip-train --resume outputs/checkpoints/deimv2_dinov3_ragwort/best_stg1.pth
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
    """Automatycznie wykrywa katalog repozytorium DEIMv2 w kontenerze lub na hoście."""
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

# Zewnętrzne zdjęcia testowe (takie same jak w YOLO)
EXTERNAL_IMAGES = [
    {
        "filename": "ragwort_field_1.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Jacobaea_vulgaris-3235.jpg",
        "expected": "ragwort",
        "description": "Starzec jakubek (Jacobaea vulgaris) kwitnacy na lace",
    },
    {
        "filename": "ragwort_flowers_2.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/%28MHNT%29_Halictus_rubicundus_on_Jacobaea_vulgaris_-_Villeneuve-les-Bouloc_France.jpg",
        "expected": "ragwort",
        "description": "Zblizenie na kwiaty starca jakubka z owadem",
    },
    {
        "filename": "negative_dandelion_3.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b5/Gesloten_bloem_van_de_paardenbloem_%28Taraxacum_officinale%29_09-05-2021._%28d.j.b%29_02.jpg",
        "expected": "negative",
        "description": "Mniszek / dmuchawiec (inny zolty kwiat - negatyw)",
    },
    {
        "filename": "negative_meadow_4.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cd/Sch%C3%B6nwald_im_Schwarzwald%2C_Escheckstra%C3%9Fe_--_2025_--_0150.jpg",
        "expected": "negative",
        "description": "Zielona laka i trawa bez starca (negatyw)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ścieżka do konfiguracji YAML DEIMv2")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu (domyślnie: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Rozmiar batcha (domyślnie: 4 dla RTX 4050 6GB)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Współczynnik uczenia detektora (domyślnie: 0.0005)")
    parser.add_argument("--workers", type=int, default=4, help="Liczba workerów DataLoader (domyślnie: 4)")
    parser.add_argument("--device", type=str, default=None, help="Urządzenie obliczeniowe: 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--good-weight", type=int, default=15, help="Waga dla danych wysokiej jakości Felix_data (domyślnie: 15x)")
    parser.add_argument("--other-weight", type=int, default=1, help="Waga dla pozostałych danych syntetycznych/combined (domyślnie: 1x)")
    parser.add_argument("--train-backbone", action="store_true", help="Odmrożenie i trenowanie backbone'a DINOv3 ViT")
    parser.add_argument("--resume", type=Path, default=None, help="Ścieżka do checkpointu .pth do wznowienia lub ewaluacji")
    parser.add_argument("--skip-train", action="store_true", help="Pomiń trening i przejdź bezpośrednio do ewaluacji i testów")
    parser.add_argument("--evaluate", action="store_true", help="Uruchom tylko ewaluację COCO (--test-only)")
    parser.add_argument("--smoke-test", action="store_true", help="Szybki test jednego batcha treningowego i walidacyjnego")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Wymuś ponowne wygenerowanie zwagowanego zbioru COCO")
    return parser.parse_args()


def ensure_weighted_coco_dataset(good_weight: int = 15, other_weight: int = 1, force: bool = False) -> Dict[str, Path]:
    """Upewnia się, że zwagowany zbiór COCO istnieje; w razie potrzeby generuje go przez weight_dataset.py."""
    train_ann = DATASETS_COCO_DIR / "train" / "annotations.json"
    val_ann = DATASETS_COCO_DIR / "val" / "annotations.json"

    if not force and train_ann.is_file() and val_ann.is_file():
        return {
            "train": train_ann,
            "val": val_ann,
            "test": DATASETS_COCO_DIR / "test" / "annotations.json",
            "img_folder": DATASETS_COCO_DIR,
        }

    print("\n[DINO_DEIM] Zwagowany zbiór COCO nie został znaleziony lub zażądano przebudowy.")
    print("[DINO_DEIM] Uruchamiam automatyczne wagowanie danych z data/data_concatenated...")
    from src.scripts.weight_dataset import process_concatenated_dataset

    process_concatenated_dataset(
        concatenated_dir=DATA_CONCAT_DIR,
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
    """Generuje listę nadpisań parametrów konfiguracji DEIMv2."""
    # Preferuj outputs/datasets/coco, w rezerwie outputs/combined_dataset_coco
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

    # Dynamiczne skalowanie epok i harmonogramu augmentacji
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
    """Oblicza IoU między dwoma ramkami [xmin, ymin, xmax, ymax]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    pole_przeciecia = (x_right - x_left) * (y_bottom - y_top)
    pole_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    pole_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    pole_calkowite = float(pole_a + pole_b - pole_przeciecia)
    return (pole_przeciecia / pole_calkowite) if pole_calkowite > 0 else 0.0


def load_deim_model(config_path: Path, checkpoint_path: Optional[Path], device: torch.device):
    """Ładuje model DEIMv2 + DINOv3 oraz postprocessor."""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt_p = Path(checkpoint_path)
        print(f"[DEIM] Ładowanie wag checkpointu: {ckpt_p}")
        data = torch.load(ckpt_p, map_location="cpu")
        state_dict = (
            data.get("ema", {}).get("module")
            or data.get("model")
            or data.get("model_state_dict")
            or data
        )
        model.load_state_dict(state_dict, strict=False)
        print("[DEIM] Pomyślnie załadowano wagi modelu!")
    return model, postprocessor


def predict_deim_single(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    conf: float = 0.25,
) -> Dict[str, Any]:
    """Wykonuje detekcję DEIMv2 na pojedynczym obrazie PIL."""
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
    """Oblicza Precision, Recall, F1 oraz zapisuje błędy FP/FN dla DEIM."""
    katalog_fp = EVAL_DIR / "bledy_False_Positives"
    katalog_fn = EVAL_DIR / "bledy_False_Negatives"
    katalog_fp.mkdir(parents=True, exist_ok=True)
    katalog_fn.mkdir(parents=True, exist_ok=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    for img_path in image_files:
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            preds = predict_deim_single(model, postprocessor, im, device, conf=conf_threshold)

        # Odczyt adnotacji GT
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        ramki_poprawne: List[List[float]] = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                p = line.strip().split()
                if len(p) >= 5 and int(p[0]) == 0:  # ragwort
                    xc, yc, bw, bh = map(float, p[1:5])
                    xmin = (xc - bw / 2.0) * orig_w
                    ymin = (yc - bh / 2.0) * orig_h
                    xmax = (xc + bw / 2.0) * orig_w
                    ymax = (yc + bh / 2.0) * orig_h
                    ramki_poprawne.append([xmin, ymin, xmax, ymax])

        ramki_modelu = preds["boxes"]
        znalezione_starce = 0

        for ramka_m in ramki_modelu:
            trafienie = False
            for ramka_p in ramki_poprawne:
                if oblicz_iou(ramka_m, ramka_p) >= prog_iou:
                    trafienie = True
                    znalezione_starce += 1
                    break

            if trafienie:
                true_positives += 1
            else:
                false_positives += 1
                try:
                    shutil.copy2(img_path, katalog_fp / f"deim_{img_path.name}")
                except Exception:
                    pass

        przegapione = len(ramki_poprawne) - znalezione_starce
        if przegapione > 0:
            false_negatives += przegapione
            try:
                shutil.copy2(img_path, katalog_fn / f"deim_{img_path.name}")
            except Exception:
                pass

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n--- WYNIKI EWALUACJI DETEKCJI DEIMv2 (IoU = 0.5) ---")
    print(f"True Positives (Trafione)        : {true_positives}")
    print(f"False Positives (Błędne alarmy)  : {false_positives}")
    print(f"False Negatives (Przegapione)    : {false_negatives}")
    print(f"Precyzja (Precision)             : {precision:.4f}")
    print(f"Czułość (Recall)                 : {recall:.4f}")
    print(f"F1-Score                         : {f1:.4f}")
    print(f"Zdjęcia z fałszywymi alarmami    : {katalog_fp}")
    print(f"Zdjęcia z przegapionymi starcami : {katalog_fn}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def pobierz_zewnetrzne_zdjecia() -> List[Path]:
    """Pobiera zdjęcia testowe z internetu."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "RagwortDetectionDEIM/1.0 (academic research; student@university.edu)"}
    sciezki = []
    print("\n--- POBIERANIE ZEWNĘTRZNYCH ZDJĘĆ Z INTERNETU ---")
    for item in EXTERNAL_IMAGES:
        cel = EXTERNAL_DIR / item["filename"]
        if not cel.exists():
            print(f"Pobieranie: {item['filename']} ({item['description']})...")
            try:
                r = requests.get(item["url"], headers=headers, timeout=20)
                if r.status_code == 200:
                    cel.write_bytes(r.content)
                    print(f"  -> Sukces ({len(r.content) // 1024} KB)")
                else:
                    print(f"  -> Błąd HTTP: {r.status_code}")
            except Exception as e:
                print(f"  -> Wyjątek: {e}")
        else:
            print(f"Plik już istnieje: {item['filename']}")
        if cel.exists():
            sciezki.append(cel)
    return sciezki


def testuj_zewnetrzne_zdjecia_deim(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    sciezki_zdjec: List[Path],
):
    """Wykonuje inferencję DEIM na zewnętrznych zdjęciach i zapisuje wizualizacje z ramkami."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_info = {item["filename"]: item for item in EXTERNAL_IMAGES}

    print("\n--- WYNIKI TESTÓW DEIMv2 NA ZEWNĘTRZNYCH ZDJĘCIACH ---")
    print(f"{'Nazwa pliku':<26} | {'Oczekiwane':<10} | {'Wykrycia':<9} | {'Max Conf':<9} | {'Werdykt'}")
    print("-" * 75)

    for img_path in sciezki_zdjec:
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
            werdykt = "SUKCES (Trafiono starca)" if n_det > 0 else "PRZEGAPIONO"
        else:
            werdykt = "SUKCES (Czysto, brak fałszywych alarmów)" if n_det == 0 else "FAŁSZYWY ALARM"

        print(f"{img_path.name:<26} | {expected:<10} | {n_det:<9} | {max_conf:<9.2f} | {werdykt}")

        # Rysowanie ramek detekcji
        draw = ImageDraw.Draw(annotated)
        for box, score, cls_name in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            color = "red" if "ragwort" in cls_name.lower() else "blue"
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)
            draw.text((xmin + 8, ymin + 8), f"{cls_name} {score:.2f}", fill="yellow")

        save_file = RESULTS_DIR / f"deim_wynik_{img_path.name}"
        annotated.save(save_file)

    print(f"\nZdjęcia z narysowanymi ramkami zapisano w: {RESULTS_DIR}")


def smoke_test(config_path: Path, overrides: list[str]) -> None:
    """Szybka walidacja jednego batcha forward / loss / backward / evaluate."""
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
        raise FloatingPointError(f"Niepoprawna strata treningowa: {loss_dict}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable_with_grad = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    frozen_with_grad = sum(p.grad is not None for p in model.backbone.dinov3.parameters() if not p.requires_grad)
    optimizer.step()
    print(f"Smoke train step: strata={loss.item():.4f}; parametry z gradientem={trainable_with_grad}; zamrożone gradienty DINOv3={frozen_with_grad}")

    evaluator = cfg.evaluator
    one_val_batch = [next(iter(cfg.val_dataloader))]
    stats, _ = evaluate(model, criterion, cfg.postprocessor.to(device), one_val_batch, evaluator, device)
    bbox = stats.get("coco_eval_bbox", [])
    if bbox:
        print(f"Smoke validation: AP={bbox[0]:.4f}, AP50={bbox[1]:.4f}, AP75={bbox[2]:.4f}")


def main() -> None:
    args = parse_args()

    # Automatyczne sprawdzenie / przygotowanie zwagowanego zbioru COCO
    ensure_weighted_coco_dataset(
        good_weight=args.good_weight,
        other_weight=args.other_weight,
        force=args.rebuild_dataset,
    )

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji: {config_path}")

    # Przejście do katalogu DEIMv2 dla zachowania poprawnych ścieżek importu
    os.chdir(DEIMV2_ROOT)
    overrides = update_items(args)

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("=" * 75)
    print(" 1. KONFIGURACJA DEIMv2 + DINOv3 (Z UWZGLĘDNIENIEM WAG)")
    print("=" * 75)
    print(f"[DEIMv2] Urządzenie:             {device}")
    if torch.cuda.is_available() and "cuda" in str(device):
        try:
            print(f"[DEIMv2] Karta GPU:              {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"[DEIMv2] Katalog DEIMv2:         {DEIMV2_ROOT}")
    print(f"[DEIMv2] Konfiguracja:           {config_path}")
    print(f"[DEIMv2] Parametry:              Epoki: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Workers: {args.workers}")
    print(f"[DEIMv2] Wagi danych:            Felix_data: {args.good_weight}x, Inne: {args.other_weight}x")

    if args.smoke_test:
        print("\n--- URUCHAMIANIE SMOKE TESTU JEDNEGO BATCHA ---")
        smoke_test(config_path, overrides)
        print("\n[SUKCES] Smoke test zakończony pomyślnie!")
        return

    output_ckpt_dir = RUNS_DIR
    best_weights_file = output_ckpt_dir / "best_stg1.pth"
    project_best_weights = MODELS_DIR / "ragwort_deimv2_best.pth"

    # --- KROK 1: TRENING ---
    if not args.skip_train and not args.evaluate:
        print("\n====================================================================")
        print(" 2. ROZPOCZĘCIE TRENINGU DEIMv2")
        print("====================================================================")
        cmd = [sys.executable, "train.py", "--config", str(config_path), "--seed", "0"]
        if overrides:
            cmd.extend(("--update", *overrides))
        print(f"[DEIMv2] Komenda: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=DEIMV2_ROOT, check=True)

        # Skopiowanie najlepszego checkpointu do katalogu models/
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        chosen_ckpt = best_weights_file if best_weights_file.exists() else (output_ckpt_dir / "last.pth")
        if chosen_ckpt.exists():
            shutil.copy2(chosen_ckpt, project_best_weights)
            print(f"\n[WAGI] Zapisano najlepsze wagi w: {project_best_weights}")

    # Wybór wag do ewaluacji
    active_weights = args.resume or project_best_weights if project_best_weights.exists() else best_weights_file
    if not active_weights or not Path(active_weights).exists():
        cand_last = output_ckpt_dir / "last.pth"
        if cand_last.exists():
            active_weights = cand_last

    # --- KROK 2: WALIDACJA COCO ---
    print("\n====================================================================")
    print(" 3. WALIDACJA COCO (mAP50, mAP50-95)")
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
            print(f"[OSTRZEŻENIE] Walidacja COCO napotkała problem: {e}")
    else:
        print("[OSTRZEŻENIE] Brak wytrenowanych wag do walidacji COCO.")

    # --- KROK 3: EWALUACJA data_eval (IoU = 0.5) ---
    print("\n====================================================================")
    print(" 4. EWALUACJA Z UŻYCIEM FUNKCJI data_eval (IoU = 0.5)")
    print("====================================================================")
    test_img_dir = DATA_CONCAT_DIR / "combined_dataset" / "test" / "images"
    test_lbl_dir = DATA_CONCAT_DIR / "combined_dataset" / "test" / "labels"

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
            print(f"[OSTRZEŻENIE] Ewaluacja data_eval napotkała błąd: {e}")
    else:
        print(f"[OSTRZEŻENIE] Katalog testowy {test_img_dir} nie istnieje.")

    # --- KROK 4: TESTY ZEWNĘTRZNE ---
    print("\n====================================================================")
    print(" 5. TESTY NA NOWYCH ZDJĘCIACH Z INTERNETU")
    print("====================================================================")
    try:
        if 'model' not in locals():
            model, postproc = load_deim_model(config_path, active_weights, device)
        zewn = pobierz_zewnetrzne_zdjecia()
        testuj_zewnetrzne_zdjecia_deim(model, postproc, device, zewn)
    except Exception as e:
        print(f"[OSTRZEŻENIE] Testy zewnętrzne pominięte: {e}")

    print("\n====================================================================")
    print(" ZAKOŃCZONO CAŁY PROCES DINO/DEIM POMYŚLNIE!")
    print("====================================================================")


if __name__ == "__main__":
    main()
