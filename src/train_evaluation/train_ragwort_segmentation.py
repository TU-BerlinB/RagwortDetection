"""
train_ragwort_segmentation.py

Skrypt do treningu, ewaluacji i wnioskowania modelu segmentacji organow starca jakubka (Ragwort).
Uczy sie wykrywac i precyzyjnie segmentowac:
  - kwiaty (flower / koszyczki kwiatowe)
  - liscie (leaf / liscie pierzastodzielne)
  - lodygi (stem)
Oraz rozpoznawac cala rosline (Ragwort) na podstawie kompozycji wysegmentowanych czesci.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Ustalenie sciezki glownej repozytorium
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.yolov8 import YOLOv8

SEG_CONFIG = REPO_ROOT / "data" / "ragwort_segmentation.yaml"
PARTS_CONFIG = REPO_ROOT / "data" / "ragwort_parts.yaml"
DEFAULT_CONFIG = SEG_CONFIG if SEG_CONFIG.exists() else PARTS_CONFIG
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


class RagwortPlantRecognizer:
    """
    Modul wnioskowania kompozycyjnego (Hierarchical / Organ-based Recognition).
    Analizuje wysegmentowane organy (kwiaty, liscie, lodygi) i ocenia,
    czy obserwowany obiekt stanowi starca jakubka (Jacobaea vulgaris)
    oraz w jakim stadium rozwoju sie znajduje (rozeta, kwitnienie, lodygowe).
    """

    ORGAN_COLORS = {
        "flower": (255, 215, 0),    # Zloty / Zolty
        "leaf": (34, 139, 34),       # Lesna zielen
        "stem": (160, 82, 45),       # Sienna / brazowawo-oliwkowy
        "ragwort": (255, 69, 0),     # Pomaranczowo-czerwony
        "default": (0, 255, 255),
    }

    def __init__(
        self,
        min_flower_conf: float = 0.25,
        min_leaf_conf: float = 0.25,
        min_stem_conf: float = 0.20,
    ):
        self.min_flower_conf = min_flower_conf
        self.min_leaf_conf = min_leaf_conf
        self.min_stem_conf = min_stem_conf

    def analyze_organs(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ocenia wystepowanie i relacje przestrzenne wysegmentowanych organow.
        """
        boxes = prediction.get("boxes", [])
        scores = prediction.get("scores", [])
        labels = prediction.get("labels", [])
        class_names = prediction.get("class_names", [])
        masks = prediction.get("masks", [])

        flowers: List[Dict[str, Any]] = []
        leaves: List[Dict[str, Any]] = []
        stems: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []

        total_flower_area = 0.0
        total_leaf_area = 0.0
        total_stem_area = 0.0

        for i, (box, score, label, name) in enumerate(zip(boxes, scores, labels, class_names)):
            name_lower = str(name).lower()
            poly = masks[i] if i < len(masks) else None
            area = cv2.contourArea(poly.astype(np.int32)) if (poly is not None and len(poly) >= 3) else (
                (box[2] - box[0]) * (box[3] - box[1]) if len(box) == 4 else 0.0
            )

            item = {
                "box": box,
                "score": score,
                "poly": poly,
                "area": area,
            }

            if "flower" in name_lower or "kwiat" in name_lower:
                if score >= self.min_flower_conf:
                    flowers.append(item)
                    total_flower_area += area
            elif "leaf" in name_lower or "lisc" in name_lower or "leaflet" in name_lower:
                if score >= self.min_leaf_conf:
                    leaves.append(item)
                    total_leaf_area += area
            elif "stem" in name_lower or "lodyg" in name_lower or "stalk" in name_lower:
                if score >= self.min_stem_conf:
                    stems.append(item)
                    total_stem_area += area
            else:
                others.append(item)

        # Reguly biometryczne rozpoznawania starca jakubka:
        is_ragwort = False
        confidence = 0.0
        stage = "unknown"

        num_flowers = len(flowers)
        num_leaves = len(leaves)
        num_stems = len(stems)

        ragwort_objs = [
            o for o in others if any("ragwort" in str(c).lower() for c in class_names)
        ]
        num_ragwort = len(ragwort_objs)
        total_ragwort_area = sum(o["area"] for o in ragwort_objs)

        if num_flowers >= 2 and (num_leaves >= 1 or num_stems >= 1):
            # Stadium pelnego kwitnienia (Flowering plant)
            is_ragwort = True
            stage = "flowering"
            flower_score = float(np.mean([f["score"] for f in flowers]))
            leaf_stem_score = float(np.mean([x["score"] for x in (leaves + stems)]))
            confidence = min(0.99, 0.6 * flower_score + 0.4 * leaf_stem_score + 0.05 * min(num_flowers, 5))
        elif num_flowers >= 1 and num_stems >= 1:
            # Wczesne kwitnienie
            is_ragwort = True
            stage = "early_flowering"
            confidence = min(0.95, 0.7 * flowers[0]["score"] + 0.3 * stems[0]["score"])
        elif num_flowers == 0 and num_leaves >= 3:
            # Stadium wegetatywne - rozeta przyziemna (Rosette)
            is_ragwort = True
            stage = "vegetative_rosette"
            avg_leaf_score = float(np.mean([l["score"] for l in leaves]))
            confidence = min(0.92, avg_leaf_score * 0.85 + 0.05 * min(num_leaves, 4))
        elif num_flowers >= 3:
            # Silne skupisko kwiatow starca
            is_ragwort = True
            stage = "flowering_inflorescence"
            confidence = float(np.mean([f["score"] for f in flowers]))
        elif len(others) > 0 and any("ragwort" in str(c).lower() for c in class_names):
            # Model uczony bezposrednio z klasa 'ragwort' (segmentacja lub detekcja calej rosliny)
            is_ragwort = True
            stage = "direct_ragwort_segmentation" if any(o.get("poly") is not None for o in ragwort_objs) else "direct_detection"
            confidence = float(np.max(scores))

        return {
            "is_ragwort": is_ragwort,
            "confidence": float(confidence),
            "stage": stage,
            "counts": {
                "flowers": num_flowers,
                "leaves": num_leaves,
                "stems": num_stems,
                "ragwort": num_ragwort,
                "total_parts": num_flowers + num_leaves + num_stems + num_ragwort,
            },
            "areas": {
                "flower_area_px": float(total_flower_area),
                "leaf_area_px": float(total_leaf_area),
                "stem_area_px": float(total_stem_area),
                "ragwort_area_px": float(total_ragwort_area),
            },
            "raw_parts": {
                "flowers": flowers,
                "leaves": leaves,
                "stems": stems,
                "ragwort": ragwort_objs,
            }
        }

    def visualize_recognition(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        prediction: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Rysuje kolorowe maski segmentacji dla kazdego organu
        oraz belke informacyjna z orzeczeniem czy roslina to starzec jakubek.
        """
        if isinstance(image_input, (str, Path)):
            cv_img = cv2.imread(str(image_input))
            if cv_img is None:
                raise FileNotFoundError(f"Nie mozna wczytac obrazu: {image_input}")
        elif isinstance(image_input, Image.Image):
            cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            cv_img = image_input.copy()

        h, w = cv_img.shape[:2]
        analysis = self.analyze_organs(prediction)

        # Warstwa masek
        overlay = cv_img.copy()

        masks = prediction.get("masks", [])
        boxes = prediction.get("boxes", [])
        class_names = prediction.get("class_names", [])
        scores = prediction.get("scores", [])

        for i, name in enumerate(class_names):
            name_l = str(name).lower()
            if "flower" in name_l:
                color = self.ORGAN_COLORS["flower"]
            elif "leaf" in name_l:
                color = self.ORGAN_COLORS["leaf"]
            elif "stem" in name_l:
                color = self.ORGAN_COLORS["stem"]
            elif "ragwort" in name_l:
                color = self.ORGAN_COLORS["ragwort"]
            else:
                color = self.ORGAN_COLORS["default"]

            # Wypelnienie poligonu maski
            if i < len(masks) and masks[i] is not None and len(masks[i]) >= 3:
                pts = masks[i].astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(cv_img, [pts], isClosed=True, color=color, thickness=2)

            # Obrys ramki
            if i < len(boxes) and len(boxes[i]) == 4:
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                score_pct = int(scores[i] * 100) if i < len(scores) else 0
                label_txt = f"{name} {score_pct}%"
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    cv_img,
                    label_txt,
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Plynne mieszanie masek (alpha blending 40%)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0, cv_img)

        # Baner naglowkowy z decyzja o rozpoznaniu starca
        header_height = 50
        header_bg = np.zeros((header_height, w, 3), dtype=np.uint8)

        if analysis["is_ragwort"]:
            header_bg[:] = (0, 100, 0)  # Ciemna zielen
            verdict_text = f"VERDICT: RAGWORT DETECTED ({analysis['confidence']*100:.1f}%) | STAGE: {analysis['stage'].upper()}"
        else:
            header_bg[:] = (0, 0, 120)  # Ciemna czerwien
            verdict_text = "VERDICT: NO RAGWORT / UNCONFIRMED"

        counts = analysis["counts"]
        details_text = f"Organs: {counts['flowers']} flowers, {counts['leaves']} leaves, {counts['stems']} stems"

        cv2.putText(header_bg, verdict_text, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header_bg, details_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        final_img = np.vstack([header_bg, cv_img])

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), final_img)
            print(f"[Recognizer] Zapisano wizualizacje weryfikacji: {out_p}")

        return final_img


def train_ragwort_segmentation(
    data: Union[str, Path] = DEFAULT_CONFIG,
    model_weight: str = "yolov8s-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.005,
    device: Optional[str] = None,
    workers: int = 2,
    project: Optional[str] = None,
    name: str = "ragwort_parts_segmentation",
    val_after: bool = True,
    demo_image: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Tuple[Any, Path]:
    """
    Glowna funkcja trenujaca model segmentacji organow starca jakubka.

    Args:
        data: Sciezka do pliku data.yaml z klasami organow (flower, leaf, stem).
        model_weight: Model bazowy (np. yolov8n-seg.pt, yolov8s-seg.pt, yolo11n-seg.pt).
        epochs: Liczba epok.
        imgsz: Rozdzielczosc wejsciowa (640, 1024).
        batch: Rozmiar partii (batch size).
        lr0: Poczatkowy wspolczynnik uczenia.
        device: 'cuda', 'cpu' lub None.
        workers: Liczba watkow data loadera.
        project: Katalog wynikow (domyslnie runs).
        name: Nazwa eksperymentu.
        val_after: Czy wykonac walidacje metryk mAP po treningu.
        demo_image: Opcjonalne zdjecie do wykonania testowej inferencji i wizualizacji.

    Returns:
        Krotka (results, best_weight_path).
    """
    data_path = Path(data)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()

    if not data_path.exists():
        print(f"[Blad] Plik konfiguracyjny {data_path} nie istnieje!")
        print("Sprawdz szablon w data/ragwort_parts.yaml i przygotuj katalogi ze zdjeciami i poligonami.")
        raise FileNotFoundError(f"Brak pliku konfiguracji zbioru: {data_path}")

    print("=" * 70)
    print(" ROZPOCZYNAM TRENING SEGMENTACJI ORGANOW STARCA JAKUBKA (RAGWORT)")
    print("=" * 70)
    print(f"Model bazowy:  {model_weight}")
    print(f"Zbiór danych:  {data_path}")
    print(f"Liczba epok:   {epochs}")
    print(f"Wymiar obrazu: {imgsz} px")
    print(f"Batch size:    {batch}")
    print(f"Urzadzenie:    {device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)

    # Inicjalizacja modelu z zadaniem 'segment'
    model = YOLOv8(model_weight=model_weight, device=device, task="segment")

    # Uruchomienie treningu
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        workers=workers,
        project=project or str(REPO_ROOT / "runs"),
        name=name,
        task="segment",
        **kwargs,
    )

    best_pt = model.best_weight_path
    if best_pt is None or not best_pt.exists():
        # Fallback sciezki wag
        candidate = REPO_ROOT / (project or "runs") / "segment" / name / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate

    print(f"\n[Trening zakonczony] Najlepsze wagi: {best_pt}")

    # Kopiowanie do models/
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_best = models_dir / "ragwort_segmentation_best.pt"
    if best_pt and best_pt.exists():
        import shutil
        shutil.copy2(best_pt, target_best)
        print(f"[Zapis] Zapisano glowny checkpoint w: {target_best}")

    # Walidacja koncowa
    if val_after and best_pt and best_pt.exists():
        print("\n--- Walidacja metryk segmentacji mAP50 i mAP50-95 ---")
        val_metrics = model.val(data=data_path, imgsz=imgsz, batch=batch)
        print("[Walidacja zakonczona pomyslnie]")

    # Demonstracyjna predykcja i wnioskowanie rozpoznania rosliny
    if demo_image:
        demo_p = Path(demo_image)
        if not demo_p.is_absolute():
            demo_p = REPO_ROOT / demo_p
        if demo_p.exists():
            print(f"\n--- Demonstracja rozpoznania na obrazie: {demo_p.name} ---")
            preds = model.predict(demo_p, conf=0.25)
            if preds:
                recognizer = RagwortPlantRecognizer()
                analysis = recognizer.analyze_organs(preds[0])
                print(f"Decyzja: {'Starzec Jakubek (Ragwort)' if analysis['is_ragwort'] else 'Brak / Niepewne'}")
                print(f"Ufnosc:  {analysis['confidence']*100:.1f}%")
                print(f"Stadium: {analysis['stage']}")
                print(f"Liczba czesci: {analysis['counts']}")

                out_vis = REPO_ROOT / "runs" / "segment" / name / f"demo_recognition_{demo_p.stem}.jpg"
                recognizer.visualize_recognition(demo_p, preds[0], output_path=out_vis)

    return results, target_best


def main():
    parser = argparse.ArgumentParser(
        description="Trening i ewaluacja modelu segmentacji organow starca (Ragwort - flower, leaf, stem)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Sciezka do data.yaml (domyslnie {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="Wagi bazowe: yolov8s-seg.pt, yolov8m-seg.pt, yolo11s-seg.pt itp.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Liczba epok treningu")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozdzielczosc obrazu (np. 640, 1024)")
    parser.add_argument("--batch", type=int, default=8, help="Rozmiar batcha")
    parser.add_argument("--lr0", type=float, default=0.005, help="Poczatkowy lr")
    parser.add_argument("--device", type=str, default=None, help="Urzadzenie: 'cuda', 'cpu' lub None")
    parser.add_argument("--workers", type=int, default=2, help="Watki danych (0 dla bezpieczenstwa na Windows)")
    parser.add_argument("--name", type=str, default="ragwort_parts_segmentation", help="Nazwa folderu uruchomienia")
    parser.add_argument("--project", type=str, default="runs", help="Katalog bazowy runs")
    parser.add_argument("--demo-image", type=str, default=None, help="Opcjonalny obraz do testowej predykcji")
    parser.add_argument("--no-val", action="store_true", help="Pominiecie walidacji po treningu")

    args = parser.parse_args()

    train_ragwort_segmentation(
        data=args.data,
        model_weight=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        val_after=not args.no_val,
        demo_image=args.demo_image,
    )


if __name__ == "__main__":
    main()
