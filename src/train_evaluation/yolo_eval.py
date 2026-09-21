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
from typing import Dict, List, Optional, Tuple, Union

import yaml

# Ścieżka bazowa projektu
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
from PIL import Image, ImageDraw
from src.models.yolov8 import YOLOv8

OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_YAML = OUTPUTS_DIR / "datasets" / "data_weighted" / "data_weighted.yaml"
RUNS_DIR = OUTPUTS_DIR / "runs" / "yolo"
MODELS_DIR = OUTPUTS_DIR / "models"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
EVAL_DIR = OUTPUTS_DIR / "evaluations" / "yolo"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions" / "yolo"

TEST_IMAGES_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "images"
TEST_LABELS_DIR = REPO_ROOT / "data" / "combined_dataset" / "test" / "labels"
EXTERNAL_DIR = REPO_ROOT / "data" / "external_test_images"
RESULTS_DIR = PREDICTIONS_DIR / "external_test_results"

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


def ensure_portable_data_yaml(yaml_path: Path) -> Path:
    """
    Weryfikuje konfigurację YAML zbioru oraz powiązane manifesty .txt,
    aby były relatywne i przenośne pomiędzy środowiskiem Docker (/workspace)
    a maszyną lokalną hosta.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return yaml_path

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[OSTRZEŻENIE] Nie można załadować {yaml_path}: {e}")
        return yaml_path

    if not isinstance(cfg, dict):
        return yaml_path

    yaml_dir = yaml_path.parent.resolve()
    modified = False

    # 1. Usuń parametr 'path' jeśli nie istnieje lub to '.' (w Ultralytics '.' resolve'uje do CWD zamiast folderu YAML)
    if "path" in cfg and cfg["path"]:
        p_val = Path(str(cfg["path"]))
        if not p_val.exists() or p_val.as_posix() == ".":
            cfg.pop("path", None)
            modified = True

    # 2. Napraw ścieżki do plików manifestów (train, val, test)
    for key in ("train", "val", "test"):
        if key in cfg and isinstance(cfg[key], str) and cfg[key].endswith(".txt"):
            txt_cand = yaml_dir / Path(cfg[key]).name
            if txt_cand.exists() and (cfg[key] != txt_cand.name or not Path(cfg[key]).exists()):
                cfg[key] = txt_cand.name
                modified = True

            # 3. Weryfikuj i napraw linie w plikach .txt jeśli zawierają bezwzględne ścieżki z innego środowiska
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
                        print(f"[INFO] Poprawiono ścieżki na relatywne w: {txt_file}")
                except Exception as e:
                    print(f"[OSTRZEŻENIE] Błąd naprawy manifestu {txt_file}: {e}")

    if modified:
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[INFO] Zaktualizowano konfigurację {yaml_path} na ścieżki relatywne.")
        except Exception as e:
            print(f"[OSTRZEŻENIE] Nie udało się zapisać poprawionego {yaml_path}: {e}")

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
    """Główna funkcja wykonująca trening (z wagami), zapis wag, ewaluację i testy zewnętrzne."""
    import torch

    # Automatyczny wybór urządzenia (CUDA / GPU lub CPU)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Automatyczne wyszukanie pliku data.yaml (priorytet dla zwagowanych danych)
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
            print("[INFO] Brak data_weighted.yaml. Uruchamiam automatyczne wagowanie...")
            from src.scripts.weight_dataset import process_concatenated_dataset
            data_yaml = process_concatenated_dataset(REPO_ROOT / "data" / "data_concatenated")

    data_yaml = ensure_portable_data_yaml(Path(data_yaml))

    # Sprawdzenie wag bazowych w outputs/weights/ lub outputs/ (aby yolov8s.pt był w outputs/)
    weights_candidate = WEIGHTS_DIR / model_name
    outputs_candidate = OUTPUTS_DIR / model_name
    if not Path(model_name).is_file():
        if weights_candidate.is_file():
            model_name = str(weights_candidate.resolve())
        elif outputs_candidate.is_file():
            model_name = str(outputs_candidate.resolve())
        elif Path(model_name).suffix in (".pt", ".pth") or not Path(model_name).parent.name:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            model_name = str((WEIGHTS_DIR / Path(model_name).name).resolve())

    print("====================================================================")
    print(" 1. TRENING MODELU YOLOV8 (Z UWZGLĘDNIENIEM WAG)")
    print("====================================================================")
    print(f"[YOLOv8] Urządzenie obliczeniowe: {device} (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and device != "cpu":
        try:
            print(f"[YOLOv8] Karta graficzna (GPU): {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"[YOLOv8] Zbiór danych:           {data_yaml}")
    print(f"[YOLOv8] Model bazowy:           {model_name}")
    print(f"[YOLOv8] Parametry:              Epoki: {epochs}, Imgsz: {imgsz}, Batch: {batch}, Cache: {cache}, Optimizer: {optimizer}, Workers: {workers}")
    print(f"[YOLOv8] Katalog wyjściowy runs: {RUNS_DIR}")

    best_weights_file = MODELS_DIR / "ragwort_yolov8_best.pt"

    if skip_train and (weights_path or best_weights_file.exists()):
        chosen_weight = weights_path or str(best_weights_file)
        print(f"[INFO] Pomijam trening. Ładuję istniejące wagi: {chosen_weight}")
        model = YOLOv8(chosen_weight, device=device)
    else:
        if not data_yaml.exists():
            raise FileNotFoundError(f"Brak pliku konfiguracji danych: {data_yaml}!")

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
        trained_best = RUNS_DIR / "ragwort_yolov8_weighted" / "weights" / "best.pt"
        if trained_best.exists():
            shutil.copy2(trained_best, best_weights_file)
            print(f"\n[WAGI] Zapisano najlepsze wagi w: {best_weights_file}")

    print("\n====================================================================")
    print(" 2. WALIDACJA MODELU NA ZBIORZE TESTOWYM (YOLO Metrics)")
    print("====================================================================")
    try:
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", device=device, verbose=False)
        print(f"mAP@50               : {metrics.box.map50:.4f}")
        print(f"mAP@50-95            : {metrics.box.map:.4f}")
        print(f"Precyzja (Precision) : {metrics.box.mp:.4f}")
        print(f"Czułość (Recall)     : {metrics.box.mr:.4f}")
    except Exception as e:
        print(f"[OSTRZEŻENIE] Nie udało się przeprowadzić standardowej walidacji YOLO: {e}")

    print("\n====================================================================")
    print(" 3. EWALUACJA Z UŻYCIEM FUNKCJI data_eval (IoU = 0.5)")
    print("====================================================================")
    # Wyznaczenie folderu testowego
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
        print(f"[data_eval] Obliczanie IoU, Precision, Recall na: {t_img}")
        ground_truth = wczytaj_poprawne_dane_z_folderu(Path(t_img), Path(t_lbl))
        predictions = model.predict_for_eval(Path(t_img), conf=0.25)
        ewaluacja_modelu(str(t_img), predictions, ground_truth, prog_iou=0.5)
    else:
        print("[OSTRZEŻENIE] Brak podzbioru testowego do ewaluacji data_eval.")

    print("\n====================================================================")
    print(" 4. TESTY NA NOWYCH ZDJĘCIACH Z INTERNETU")
    print("====================================================================")
    try:
        zewn_zdjecia = pobierz_zewnetrzne_zdjecia()
        testuj_zewnetrzne_zdjecia(model, zewn_zdjecia)
    except Exception as e:
        print(f"[OSTRZEŻENIE] Testy zewnętrzne pominięte: {e}")

    print("\n====================================================================")
    print(" ZAKOŃCZONO CAŁY PROCES POMYŚLNIE!")
    print("====================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="data_eval - Trening YOLOv8 (z wagami) i ewaluacja detekcji.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=None, help="Ścieżka do data.yaml (domyślnie auto-wykrywanie data_weighted.yaml)")
    parser.add_argument("--device", type=str, default=None, help="Urządzenie: '0' (CUDA/GPU), 'cpu' (domyślnie: auto-wykrycie CUDA)")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Wagi modelu bazowego (domyślnie: yolov8s.pt - Small)")
    parser.add_argument("--epochs", type=int, default=50, help="Liczba epok treningu (domyślnie: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozdzielczość obrazu (domyślnie: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha (domyślnie: 16, zmniejsz do 8 przy małym VRAM)")
    parser.add_argument("--workers", type=int, default=8, help="Liczba wątków loadera danych (domyślnie: 8 dla procesora 6c/12t)")
    parser.add_argument("--cache", type=str, default="none", choices=["none", "ram", "disk"], help="Cache obrazów: 'ram' (w pamięci RAM), 'disk' lub 'none' (domyślnie)")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam", "auto"], help="Optymalizator (domyślnie: AdamW, stabilny i bezpieczny na GPU)")
    parser.add_argument("--skip-train", action="store_true", help="Pomiń trening i załaduj istniejące wagi")
    parser.add_argument("--weights", type=str, default=None, help="Własna ścieżka do wag .pt")
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