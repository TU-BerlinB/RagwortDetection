"""
data_eval.py

Skrypt do treningu, ewaluacji i testowania modelu detekcji starca (RagwortDetection).

Zawiera:
  1. Obliczanie IoU (oblicz_iou) i ewaluację detekcji (ewaluacja_modelu: Precision, Recall, FP, FN).
  2. Trening modelu YOLOv8 na danych z data/combined_dataset (wszystkie dane treningowe).
  3. Zapisanie najlepszych wag do models/ragwort_yolov8_best.pt.
  4. Testy walidacyjne na podzbiorze testowym (mAP50, mAP50-95).
  5. Pobranie zewnętrznych zdjęć z internetu (starzec jakubek, inne żółte kwiaty, łąka).
  6. Test poprawności na nowych danych zewnętrznych i zapis wizualizacji z ramkami.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ścieżka bazowa projektu
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
from PIL import Image, ImageDraw
from src.models.yolov8 import YOLOv8

DATA_YAML = REPO_ROOT / "data" / "combined_dataset" / "data.yaml"
TEST_IMAGES_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "images"
TEST_LABELS_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "labels"
MODELS_DIR = REPO_ROOT / "models"
EXTERNAL_DIR = REPO_ROOT / "data" / "external_test_images"
RESULTS_DIR = REPO_ROOT / "data" / "external_test_results"

# Zewnętrzne zdjęcia z internetu (Wikimedia Commons) do weryfikacji generalizacji
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


def oblicz_iou(box_a, box_b):
    """
    Oblicza IoU [Intersection over Union - procent pokrycia się dwóch ramek].
    Format ramek: [x_min, y_min, x_max, y_max]
    """
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
    if pole_calkowite <= 0:
        return 0.0

    return pole_przeciecia / pole_calkowite


def ewaluacja_modelu(katalog_zdjec, predykcje, poprawne_dane, prog_iou=0.5):
    """
    Główna funkcja oceniająca model pod kątem IoU, Precision, Recall oraz błędów FP/FN.
    """
    katalog_fp = REPO_ROOT / "bledy_False_Positives"
    katalog_fn = REPO_ROOT / "bledy_False_Negatives"

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

    print("--- WYNIKI EWALUACJI (data_eval) ---")
    print(f"True Positives (Trafione)        : {true_positives}")
    print(f"False Positives (Błędne alarmy)  : {false_positives}")
    print(f"False Negatives (Przegapione)    : {false_negatives}")
    print(f"Precyzja (Precision)             : {precision:.4f}")
    print(f"Czułość (Recall)                 : {recall:.4f}")
    print(f"F1-Score                         : {f1:.4f}")
    print(f"Zdjęcia z fałszywymi alarmami zapisano w : {katalog_fp}")
    print(f"Zdjęcia z przegapionymi starcami zapisano w: {katalog_fn}")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def wczytaj_poprawne_dane_z_folderu(images_dir: Path, labels_dir: Path) -> Dict[str, List[List[float]]]:
    """Wczytuje etykiety Ground Truth z plików YOLO .txt i przelicza na format pikseli [xmin, ymin, xmax, ymax]."""
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
                        # Klasa 0 to starzec (ragwort)
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
    """Pobiera zewnętrzne zdjęcia z internetu (Wikimedia Commons) do weryfikacji."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "RagwortDetectionResearch/1.0 (academic research; student@university.edu)"}
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
            print(f"Plik już istnieje lokalnie: {item['filename']}")

        if cel.exists():
            sciezki.append(cel)

    return sciezki


def testuj_zewnetrzne_zdjecia(model: YOLOv8, sciezki_zdjec: List[Path]):
    """Wykonuje inferencję na zewnętrznych zdjęciach i zapisuje wyniki z ramkami."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_info = {item["filename"]: item for item in EXTERNAL_IMAGES}

    print("\n--- WYNIKI TESTÓW NA ZEWNĘTRZNYCH ZDJĘCIACH ---")
    print(f"{'Nazwa pliku':<26} | {'Oczekiwane':<10} | {'Wykrycia':<9} | {'Max Conf':<9} | {'Werdykt'}")
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
            werdykt = "SUKCES (Trafiono starca)" if n_det > 0 else "PRZEGAPIONO"
        else:
            werdykt = "SUKCES (Czysto, brak fałszywych alarmów)" if n_det == 0 else "FAŁSZYWY ALARM"

        print(f"{img_path.name:<26} | {expected:<10} | {n_det:<9} | {max_conf:<9.2f} | {werdykt}")

        # Rysowanie ramek
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

    print(f"\nZdjęcia z narysowanymi ramkami zapisano w katalogu: {RESULTS_DIR}")


def trenuj_i_ewaluuj(
    epochs: int = 5,
    imgsz: int = 416,
    batch: int = 16,
    skip_train: bool = False,
    weights_path: Optional[str] = None,
):
    """Główna funkcja wykonująca trening, zapis wag, ewaluację i testy zewnętrzne."""
    print("====================================================================")
    print(" 1. TRENING MODELU YOLOV8")
    print("====================================================================")

    best_weights_file = MODELS_DIR / "ragwort_yolov8_best.pt"

    if skip_train and (weights_path or best_weights_file.exists()):
        chosen_weight = weights_path or str(best_weights_file)
        print(f"[INFO] Pomijam trening. Ładuję istniejące wagi: {chosen_weight}")
        model = YOLOv8(chosen_weight)
    else:
        if not DATA_YAML.exists():
            raise FileNotFoundError(f"Brak pliku {DATA_YAML}. Uruchom najpierw data_download.py!")

        model = YOLOv8("yolov8n.pt")
        print(f"[YOLOv8] Urządzenie obliczeniowe: {model.device}")
        print(f"[YOLOv8] Rozpoczynam trening na wszystkich danych: {DATA_YAML} (Epoki: {epochs}, Imgsz: {imgsz}, Batch: {batch})")

        model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="ragwort_yolov8",
            exist_ok=True,
            verbose=True,
        )

    print(f"\n[WAGI] Zapisano najlepsze wagi w: {best_weights_file}")

    print("\n====================================================================")
    print(" 2. WALIDACJA MODELU NA ZBIORZE TESTOWYM (YOLO Metrics)")
    print("====================================================================")
    metrics = model.val(data=DATA_YAML, imgsz=imgsz, split="val", verbose=False)
    print(f"mAP@50               : {metrics.box.map50:.4f}")
    print(f"mAP@50-95            : {metrics.box.map:.4f}")
    print(f"Precyzja (Precision) : {metrics.box.mp:.4f}")
    print(f"Czułość (Recall)     : {metrics.box.mr:.4f}")

    print("\n====================================================================")
    print(" 3. EWALUACJA Z UŻYCIEM FUNKCJI data_eval (IoU = 0.5)")
    print("====================================================================")
    if TEST_IMAGES_DIR.exists() and TEST_LABELS_DIR.exists():
        ground_truth = wczytaj_poprawne_dane_z_folderu(TEST_IMAGES_DIR, TEST_LABELS_DIR)
        predictions = model.predict_for_eval(TEST_IMAGES_DIR, conf=0.25)
        ewaluacja_modelu(str(TEST_IMAGES_DIR), predictions, ground_truth, prog_iou=0.5)
    else:
        print("[OSTRZEŻENIE] Brak podzbioru testowego do ewaluacji data_eval.")

    print("\n====================================================================")
    print(" 4. TESTY NA NOWYCH ZDJĘCIACH Z INTERNETU")
    print("====================================================================")
    zewn_zdjecia = pobierz_zewnetrzne_zdjecia()
    testuj_zewnetrzne_zdjecia(model, zewn_zdjecia)

    print("\n====================================================================")
    print(" ZAKOŃCZONO CAŁY PROCES POMYŚLNIE!")
    print("====================================================================")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5, help="Liczba epok treningu (domyślnie: 5)")
    parser.add_argument("--imgsz", type=int, default=416, help="Rozdzielczość obrazu wejściowego (domyślnie: 416)")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha (domyślnie: 16)")
    parser.add_argument("--skip-train", action="store_true", help="Pomiń trening i załaduj wagi z models/ragwort_yolov8_best.pt")
    parser.add_argument("--weights", type=str, default=None, help="Własna ścieżka do wag .pt")
    args = parser.parse_args()

    trenuj_i_ewaluuj(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        skip_train=args.skip_train,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()