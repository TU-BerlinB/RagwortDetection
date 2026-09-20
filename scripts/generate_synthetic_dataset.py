"""
generate_synthetic_dataset.py
Skrypt do syntetycznego generowania fotorealistycznego zbioru danych (900 obrazów)
przedstawiających wycinek łąki 1.5m x 1.5m z pojedynczymi roślinami starca (ragwort).
Zaprojektowany specjalnie pod wymagania kamery robota rolniczego (np. Hege / Framos D435e).
"""

import os
import sys
import glob
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic_meadow_150cm"

# Skala fizyczna:
# Pole widzenia (FOV): 1.5m x 1.5m (150 cm x 150 cm)
# Rozdzielczość obrazu: 1024 x 1024 px -> 1 px = ok. 1.46 mm
# Rozmiar pojedynczej rośliny starca na łące:
# - mała rozeta: 8-15 cm (ok. 55 - 105 px)
# - średnia roślina: 15-25 cm (ok. 105 - 175 px)
# - dojrzały kwitnący starzec: 25-40 cm (ok. 175 - 275 px)
IMAGE_SIZE = 1024


def extract_ragwort_sprites() -> List[np.ndarray]:
    """Wyodrębnia wycięte rozety/rośliny starca z kanałem przezroczystości RGBA."""
    print("[1/4] Wyodrębnianie wycinków roślin starca z poligonów referencyjnych...")
    label_files = glob.glob(str(PROJECT_ROOT / "data" / "combined_dataset" / "*" / "labels" / "*.txt"))
    seg_files = [f for f in label_files if any(len(l.split()) > 5 for l in open(f).readlines())]

    sprites = []
    for f in seg_files:
        img_f = f.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(img_f):
            img_f = f.replace("labels", "images").replace(".txt", ".png")
        if not os.path.exists(img_f):
            continue

        img = cv2.imread(img_f)
        if img is None:
            continue

        h, w = img.shape[:2]
        lines = open(f).readlines()
        for line in lines:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) > 5:
                pts = np.array([[int(parts[i]*w), int(parts[i+1]*h)] for i in range(1, len(parts), 2)], dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                if bw < 25 or bh < 25:
                    continue

                # Odrzucenie poligonów, które dotykają krawędzi kadru (są sztucznymi kwadratowymi wycinkami)
                touch_left = np.any(pts[:, 0] <= 2)
                touch_right = np.any(pts[:, 0] >= w - 3)
                touch_top = np.any(pts[:, 1] <= 2)
                touch_bottom = np.any(pts[:, 1] >= h - 3)
                if sum([touch_left, touch_right, touch_top, touch_bottom]) >= 3:
                    continue

                area = cv2.contourArea(pts)
                extent = area / (bw * bh) if (bw * bh) > 0 else 0
                if extent > 0.78 and (bw / float(w)) > 0.85:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                plant_crop = img[y:y+bh, x:x+bw]
                mask_crop = mask[y:y+bh, x:x+bw]

                # Wygładzenie krawędzi (feathering) zapobiega ostrym sztucznym obwódkom
                mask_blur = cv2.GaussianBlur(mask_crop, (7, 7), 0)

                rgba = cv2.cvtColor(plant_crop, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask_blur
                sprites.append(rgba)

    print(f" -> Załadowano {len(sprites)} autentycznych wycinków starca jakubka.")
    return sprites


def load_meadow_sources() -> List[np.ndarray]:
    """Wczytuje wysokorozdzielcze zdjęcia autentycznych łąk i pastwisk (widok z góry) jako źródła tła."""
    print("[2/4] Wczytywanie tekstur i zdjęć łąki z góry (top-down pasture/meadow)...")
    sources = []

    bg_dir = PROJECT_ROOT / "data" / "meadow_backgrounds"
    bg_files = glob.glob(str(bg_dir / "*.*"))

    for p in bg_files:
        if p.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(p)
            if img is not None:
                sources.append(img)
                print(f" -> Załadowano tło łąkowe {Path(p).name}: {img.shape[1]}x{img.shape[0]} px")

    print(f" -> Załadowano łącznie {len(sources)} autentycznych wysokorozdzielczych teł łąki.")
    return sources


def get_random_meadow_patch(sources: List[np.ndarray], target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Wycina losowy fragment łąki odpowiadający obszarowi 1.5m x 1.5m i aplikuje zróżnicowane oświetlenie."""
    src = random.choice(sources)
    sh, sw = src.shape[:2]

    # Określenie rozmiaru wycinka (odpowiadającego 1.5m x 1.5m w skali zdjęcia źródłowego)
    min_dim = min(sh, sw)
    crop_size = random.randint(min(target_size, min_dim), min_dim)

    max_y = sh - crop_size
    max_x = sw - crop_size
    y = random.randint(0, max_y) if max_y > 0 else 0
    x = random.randint(0, max_x) if max_x > 0 else 0

    patch = src[y:y+crop_size, x:x+crop_size].copy()
    if patch.shape[0] != target_size or patch.shape[1] != target_size:
        patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Losowe obroty i odbicia (dla pełnej różnorodności kierunku źdźbeł trawy)
    if random.random() > 0.5:
        patch = cv2.flip(patch, 1)  # Odbicie lustrzane
    if random.random() > 0.5:
        patch = cv2.flip(patch, 0)
    rot = random.choice([0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot != 0:
        patch = cv2.rotate(patch, rot)

    # Urozmaicenie warunków oświetleniowych:
    # 1. Jasność i kontrast (słońce / chmury / wieczór)
    alpha = random.uniform(0.85, 1.15)  # Kontrast
    beta = random.randint(-20, 20)       # Jasność
    patch = np.clip(alpha * patch.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 2. Symulacja naturalnego cienia chmur lub koron drzew (gładki gradient oświetlenia)
    if random.random() > 0.4:
        # Generowanie łagodnego cienia o niskiej częstotliwości
        small_noise = np.random.uniform(0.70, 1.0, (8, 8)).astype(np.float32)
        shadow_map = cv2.resize(small_noise, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        shadow_map = cv2.GaussianBlur(shadow_map, (101, 101), 0)
        patch = np.clip(patch.astype(np.float32) * shadow_map[:, :, None], 0, 255).astype(np.uint8)

    return patch


def rotate_and_scale_sprite(sprite: np.ndarray, target_px: int, angle_deg: float) -> np.ndarray:
    """Skaluje i obraca wycinek rośliny bez przycinania brzegów."""
    sh, sw = sprite.shape[:2]
    scale = target_px / max(sh, sw)
    nw = max(10, int(sw * scale))
    nh = max(10, int(sh * scale))
    resized = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Obrót wokół środka z rozszerzeniem ramki
    center = (nw // 2, nh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = int((nh * sin_val) + (nw * cos_val))
    new_h = int((nh * cos_val) + (nw * sin_val))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(resized, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


def place_plant_on_meadow(
    meadow: np.ndarray,
    sprite: np.ndarray,
    cx: int,
    cy: int,
    target_px: int,
    angle_deg: float
) -> Tuple[np.ndarray, List[float]]:
    """Umieszcza pojedynczą roślinę na łące z cieniem gruntowym i wylicza ramkę YOLO."""
    transformed = rotate_and_scale_sprite(sprite, target_px, angle_deg)
    th, tw = transformed.shape[:2]

    # Obliczenie współrzędnych wklejenia
    x1 = cx - tw // 2
    y1 = cy - th // 2
    x2 = x1 + tw
    y2 = y1 + th

    # Obcięcie do granic obrazu
    img_h, img_w = meadow.shape[:2]
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return meadow, []

    spr_x1 = crop_x1 - x1
    spr_y1 = crop_y1 - y1
    spr_x2 = spr_x1 + (crop_x2 - crop_x1)
    spr_y2 = spr_y1 + (crop_y2 - crop_y1)

    spr_patch = transformed[spr_y1:spr_y2, spr_x1:spr_x2]
    alpha = spr_patch[:, :, 3].astype(np.float32) / 255.0

    # Sprawdzenie czy widać roślinę
    if np.count_nonzero(alpha > 0.15) < 30:
        return meadow, []

    # 1. Naturalny cień gruntowy (delikatne przyciemnienie trawy bezpośrednio pod liśćmi)
    shadow_offset_x = random.randint(3, 8)
    shadow_offset_y = random.randint(4, 10)
    sh_x1 = max(0, crop_x1 + shadow_offset_x)
    sh_y1 = max(0, crop_y1 + shadow_offset_y)
    sh_x2 = min(img_w, crop_x2 + shadow_offset_x)
    sh_y2 = min(img_h, crop_y2 + shadow_offset_y)

    if (sh_x2 > sh_x1) and (sh_y2 > sh_y1):
        sh_spr_w = sh_x2 - sh_x1
        sh_spr_h = sh_y2 - sh_y1
        shadow_mask = cv2.GaussianBlur(alpha[:sh_spr_h, :sh_spr_w], (15, 15), 0) * 0.35
        for c in range(3):
            meadow[sh_y1:sh_y2, sh_x1:sh_x2, c] = np.clip(
                meadow[sh_y1:sh_y2, sh_x1:sh_x2, c].astype(np.float32) * (1.0 - shadow_mask),
                0, 255
            ).astype(np.uint8)

    # 2. Dopasowanie odcienia / tonacji rośliny do lokalnego tła trawy
    bg_roi = meadow[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    spr_rgb = spr_patch[:, :, :3].astype(np.float32)

    # Subtelny jitter jasności rośliny (+/- 10%)
    color_jitter = random.uniform(0.92, 1.08)
    spr_rgb = np.clip(spr_rgb * color_jitter, 0, 255)

    # 3. Wklejenie rośliny metodą miękkiego kanału alfa (alpha blending)
    alpha_3d = alpha[:, :, None]
    blended_roi = spr_rgb * alpha_3d + bg_roi * (1.0 - alpha_3d)
    meadow[crop_y1:crop_y2, crop_x1:crop_x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    # 4. Wyznaczenie dokładnego Bounding Boxa w formacie YOLO
    non_zero = np.argwhere(alpha > 0.15)
    if len(non_zero) == 0:
        return meadow, []

    ymin_local, xmin_local = non_zero.min(axis=0)
    ymax_local, xmax_local = non_zero.max(axis=0)

    box_xmin = crop_x1 + xmin_local
    box_xmax = crop_x1 + xmax_local
    box_ymin = crop_y1 + ymin_local
    box_ymax = crop_y1 + ymax_local

    bw = box_xmax - box_xmin
    bh = box_ymax - box_ymin

    if bw < 15 or bh < 15:
        return meadow, []

    # Znormalizowane koordynaty YOLO [0.0 - 1.0]
    bx_center = (box_xmin + box_xmax) / (2.0 * img_w)
    by_center = (box_ymin + box_ymax) / (2.0 * img_h)
    norm_w = bw / float(img_w)
    norm_h = bh / float(img_h)

    # Ograniczenie wartości do zakresu [0.0, 1.0]
    bx_center = min(max(bx_center, 0.0), 1.0)
    by_center = min(max(by_center, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.005), 1.0)
    norm_h = min(max(norm_h, 0.005), 1.0)

    return meadow, [bx_center, by_center, norm_w, norm_h]


def generate_synthetic_dataset(num_images: int = 900):
    """Generuje kompletny zbiór 900 obrazów z etykietami YOLO i podziałem train/val."""
    start_time = time.time()
    print("===================================================================")
    print(f" ROZPOCZYNAM GENEROWANIE {num_images} OBRAZÓW ŁĄKI 1.5m x 1.5m Z RAGWORT")
    print(f" Cel: Zbiór idealny do trenowania modelu detekcji chwastów YOLO")
    print(f" Folder docelowy: {OUTPUT_DATASET_DIR}")
    print("===================================================================")

    # Przygotowanie katalogów
    train_img_dir = OUTPUT_DATASET_DIR / "train" / "images"
    train_lbl_dir = OUTPUT_DATASET_DIR / "train" / "labels"
    val_img_dir = OUTPUT_DATASET_DIR / "val" / "images"
    val_lbl_dir = OUTPUT_DATASET_DIR / "val" / "labels"
    preview_dir = OUTPUT_DATASET_DIR / "preview_visualizations"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, preview_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sprites = extract_ragwort_sprites()
    sources = load_meadow_sources()

    if len(sprites) == 0:
        raise RuntimeError("Brak wycinków roślin starca! Sprawdź ścieżkę do danych.")
    if len(sources) == 0:
        raise RuntimeError("Brak zdjęć łąki! Sprawdź ścieżkę do obrazów tła.")

    # Podział: 750 trening (ok. 83.3%), 150 walidacja (ok. 16.7%)
    num_train = int(num_images * (750 / 900))
    print(f"\n[3/4] Generowanie {num_images} obrazów ({num_train} train, {num_images - num_train} val)...")

    total_plants = 0
    saved_previews = 0

    for i in range(num_images):
        is_val = (i >= num_train)
        img_dir = val_img_dir if is_val else train_img_dir
        lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        base_name = f"meadow_150cm_{i:04d}"
        img_path = img_dir / f"{base_name}.jpg"
        lbl_path = lbl_dir / f"{base_name}.txt"

        # 1. Wycinek łąki 1.5m x 1.5m
        meadow = get_random_meadow_patch(sources, target_size=IMAGE_SIZE)

        # 2. Liczba pojedynczych roślin w kadrze 1.5m:
        # - 65% szans: dokładnie 1 pojedyncza roślina (zgodnie z życzeniem użytkownika)
        # - 20% szans: 2 pojedyncze rośliny w różnych częściach kadru
        # - 10% szans: 3 pojedyncze rośliny
        # - 5% szans: 0 roślin (negatywne tło łąki, kluczowe dla redukcji fałszywych alarmów)
        dice = random.random()
        if dice < 0.05:
            num_plants = 0
        elif dice < 0.70:
            num_plants = 1
        elif dice < 0.90:
            num_plants = 2
        else:
            num_plants = 3

        boxes = []
        occupied_centers = []

        for _ in range(num_plants):
            sprite = random.choice(sprites)

            # Fizyczny rozmiar rośliny w skali 1.5m (od 60px do 240px)
            # Rozkład naturalny: najwięcej średnich rozet (100-160px), rzadziej małe siewki lub wielkie kępy
            plant_px = int(np.random.normal(loc=135, scale=40))
            plant_px = max(60, min(plant_px, 250))

            # Szukanie wolnego miejsca na łące (żeby rośliny nie nachodziły na siebie)
            margin = plant_px // 2 + 30
            placed = False
            for attempt in range(25):
                cx = random.randint(margin, IMAGE_SIZE - margin)
                cy = random.randint(margin, IMAGE_SIZE - margin)

                # Dystans od już umieszczonych roślin
                too_close = False
                for ox, oy in occupied_centers:
                    dist = math.hypot(cx - ox, cy - oy)
                    if dist < (plant_px + 70):
                        too_close = True
                        break

                if not too_close:
                    occupied_centers.append((cx, cy))
                    placed = True
                    break

            if not placed:
                continue

            angle = random.uniform(0, 360)
            meadow, yolo_box = place_plant_on_meadow(meadow, sprite, cx, cy, plant_px, angle)
            if len(yolo_box) == 4:
                boxes.append(yolo_box)

        # Zapis etykiety YOLO (.txt)
        with open(lbl_path, "w") as f:
            for b in boxes:
                # Klasa 0: ragwort (starzec jakubek)
                f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        # Zapis obrazu (.jpg o wysokiej jakości)
        cv2.imwrite(str(img_path), meadow, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        total_plants += len(boxes)

        # Zapis kilku wizualizacji weryfikacyjnych (z narysowanymi ramkami)
        if saved_previews < 8 and len(boxes) > 0:
            preview_img = meadow.copy()
            for b in boxes:
                bx, by, bw, bh = b
                xmin = int((bx - bw/2) * IMAGE_SIZE)
                ymin = int((by - bh/2) * IMAGE_SIZE)
                xmax = int((bx + bw/2) * IMAGE_SIZE)
                ymax = int((by + bh/2) * IMAGE_SIZE)

                cv2.rectangle(preview_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.circle(preview_img, (int(bx * IMAGE_SIZE), int(by * IMAGE_SIZE)), 5, (0, 255, 255), -1)
                cv2.putText(preview_img, "ragwort 1.5m", (xmin, max(25, ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imwrite(str(preview_dir / f"preview_{saved_previews:02d}.jpg"), preview_img)
            saved_previews += 1

        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            elapsed = time.time() - start_time
            print(f" -> Wygenerowano {i + 1}/{num_images} obrazów (Łącznie roślin: {total_plants}, Czas: {elapsed:.1f}s)")

    # 4. Utworzenie pliku konfiguracyjnego dataset.yaml
    print("\n[4/4] Tworzenie konfiguracji data.yaml do treningu YOLO...")
    data_yaml_path = OUTPUT_DATASET_DIR / "data.yaml"
    yaml_content = f"""# Konfiguracja zbioru danych: 1.5m x 1.5m Synthetic Meadow with Ragwort
# Wygenerowano 900 obrazów (750 train, 150 val)
path: {OUTPUT_DATASET_DIR.as_posix()}
train: train/images
val: val/images

nc: 1
names:
  0: ragwort
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    total_time = time.time() - start_time
    print("===================================================================")
    print(f" SUKCES! ZBIÓR DANYCH ZOSTAŁ POMYŚLNIE WYGENEROWANY!")
    print(f" Liczba wygenerowanych obrazów: {num_images}")
    print(f" Liczba wygenerowanych etykiet YOLO: {num_images}")
    print(f" Łączna liczba umieszczonych roślin starca: {total_plants}")
    print(f" Średnia liczba roślin na kadr 1.5m x 1.5m: {total_plants / num_images:.2f}")
    print(f" Przykładowe wizualizacje z ramkami: {preview_dir}")
    print(f" Plik do trenowania YOLO: {data_yaml_path}")
    print(f" Całkowity czas generowania: {total_time:.1f} s")
    print("===================================================================")


if __name__ == "__main__":
    count = 900
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    generate_synthetic_dataset(count)
