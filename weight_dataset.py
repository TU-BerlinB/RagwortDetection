"""
weight_dataset.py - Skrypt do automatycznego wagowania danych dla RagwortDetection.

Obsluguje 2 tryby dzialania:
TRYB 1: data_concatenated (Twoja struktura na SzymonPC / Linux):
  ├── Felix_data/
  │   └── label_200/label/
  │       ├── annotation/             <- etykiety (.txt lub .xml)
  │       └── images_selected_200/    <- DANE DOBRE (GOOD) -> waga np. 15x
  ├── combined_dataset/
  │   ├── train/ (images, labels)     <- DANE POZOSTALE (OTHER) -> waga 1x
  │   └── test/  (images, labels)     <- ZBIOR TESTOWY / VAL
  ├── synthetic/ (images, labels)     <- DANE POZOSTALE (OTHER) -> waga 1x
  └── synthetic_dataset_split/
      ├── part_4/ (images, labels)    <- DANE POZOSTALE (OTHER) -> waga 1x
      └── part1images/ (images)       <- DANE POZOSTALE (OTHER) -> waga 1x

TRYB 2: Foldery good/ i other/ (reczne wrzucanie zdjec):
  - good/   -> zdjecia idealne
  - other/  -> pozostale zdjecia

Zalety:
1. Zero kopiowania gigabajtow zdjec! Tworzy relatywny train_weighted.txt i data_weighted.yaml.
2. Automatycznie konwertuje adnotacje .xml (Pascal VOC) do .txt (YOLO) dla Felix_data jesli to konieczne.
3. Gwarantuje, ze kazde zdjecie tla ma poprawny (pusty) plik .txt, dzieki czemu YOLO sie nie wywali.
4. Generuje plik data_weighted.yaml, ktory mozna od razu przekazac do data_eval.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def convert_voc_xml_to_yolo_txt(xml_path: Path, output_txt: Path) -> bool:
    """Konwertuje etykiety z formatu Pascal VOC XML na format YOLO TXT (znormalizowany)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            return False
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        if width <= 0 or height <= 0:
            return False

        yolo_lines = []
        for obj in root.findall("object"):
            name = (obj.find("name").text or "").lower()
            cls_id = 0 if any(k in name for k in ["ragwort", "senecio", "chwast", "jakob"]) else 1

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Obliczenie wspolrzednych YOLO
            xc = ((xmin + xmax) / 2.0) / width
            yc = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Ograniczenie do [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        output_txt.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        output_txt.write_text("", encoding="utf-8")
        return False


def ensure_label_exists_for_image(img_path: Path, possible_label_dirs: List[Path]) -> Path:
    """
    Zapewnia, ze dla danego zdjecia istnieje plik etykiety .txt w miejscu,
    gdzie YOLO go znajdzie (obok zdjecia lub w labels/).
    """
    stem = img_path.stem

    # 1. Sprawdz czy obok zdjecia jest niepusty plik .txt
    neighbor_txt = img_path.parent / f"{stem}.txt"
    if neighbor_txt.is_file() and neighbor_txt.stat().st_size > 0:
        return neighbor_txt

    # 2. Sprawdz czy w folderze nadrzednym w labels/ jest niepusty .txt
    sub_lbl = img_path.parent.parent / "labels" / f"{stem}.txt"
    if sub_lbl.is_file() and sub_lbl.stat().st_size > 0:
        return sub_lbl

    # 3. Przeszukaj liste kandydatow katalogow
    for d in possible_label_dirs:
        if not d.exists():
            continue

        txt_cand = d / f"{stem}.txt"
        if txt_cand.is_file() and txt_cand.stat().st_size > 0:
            try:
                shutil.copy2(txt_cand, neighbor_txt)
                return neighbor_txt
            except Exception:
                return txt_cand

        xml_cand = d / f"{stem}.xml"
        if xml_cand.is_file() and xml_cand.stat().st_size > 0:
            if convert_voc_xml_to_yolo_txt(xml_cand, neighbor_txt):
                return neighbor_txt

        # Sprawdz pliki ignorujac wielkosc liter (case-insensitive)
        for f in d.rglob("*"):
            if f.is_file() and f.stem.lower() == stem.lower():
                if f.suffix.lower() == ".txt" and f.stat().st_size > 0:
                    try:
                        shutil.copy2(f, neighbor_txt)
                        return neighbor_txt
                    except Exception:
                        return f
                elif f.suffix.lower() == ".xml" and f.stat().st_size > 0:
                    if convert_voc_xml_to_yolo_txt(f, neighbor_txt):
                        return neighbor_txt

    # 4. Jesli zupelnie brak etykiety -> utworz pusty plik (obraz tla / negatyw)
    if not neighbor_txt.is_file():
        neighbor_txt.write_text("", encoding="utf-8")
    return neighbor_txt


def process_concatenated_dataset(
    concatenated_dir: Path,
    good_weight: int = 15,
    other_weight: int = 1,
    val_felix_ratio: float = 0.10,
) -> Path:
    """
    Wagowanie bezposrednio w strukturze data_concatenated:
    - Felix_data -> GOOD (waga np. 15x)
    - combined_dataset/train, synthetic, synthetic_dataset_split -> OTHER (waga 1x)
    - combined_dataset/test -> VAL
    """
    concatenated_dir = Path(concatenated_dir).resolve()
    print("=" * 75)
    print(" WAGOWANIE ZBIORU DANYCH W TRYBIE: data_concatenated")
    print(f" Katalog bazowy: {concatenated_dir}")
    print("=" * 75)

    felix_images: List[Path] = []
    other_images: List[Path] = []
    val_images: List[Path] = []

    # 1. Wyszukaj dane Felix_data (GOOD)
    felix_dir = concatenated_dir / "Felix_data"
    felix_label_dirs = []
    if felix_dir.exists():
        for d in felix_dir.rglob("*"):
            if d.is_dir() and any(k in d.name.lower() for k in ["annot", "label"]):
                felix_label_dirs.append(d)

        # Znajdz zdjecia w images_selected_200 lub w Felix_data
        for p in felix_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                felix_images.append(p)
                ensure_label_exists_for_image(p, felix_label_dirs)

    # 2. Wyszukaj dane walidacyjne / testowe (combined_dataset/test)
    test_dir = concatenated_dir / "combined_dataset" / "test" / "images"
    test_label_dirs = [concatenated_dir / "combined_dataset" / "test" / "labels"]
    if test_dir.exists():
        for p in test_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                val_images.append(p)
                ensure_label_exists_for_image(p, test_label_dirs)

    # 3. Wyszukaj dane pozostale (OTHER)
    # 3a. combined_dataset/train
    comb_train = concatenated_dir / "combined_dataset" / "train" / "images"
    comb_train_lbls = [concatenated_dir / "combined_dataset" / "train" / "labels"]
    if comb_train.exists():
        for p in comb_train.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                other_images.append(p)
                ensure_label_exists_for_image(p, comb_train_lbls)

    # 3b. synthetic
    synth_dir = concatenated_dir / "synthetic" / "images"
    synth_lbls = [concatenated_dir / "synthetic" / "labels"]
    if synth_dir.exists():
        for p in synth_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                other_images.append(p)
                ensure_label_exists_for_image(p, synth_lbls)

    # 3c. synthetic_dataset_split
    split_dir = concatenated_dir / "synthetic_dataset_split"
    if split_dir.exists():
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                # Jesli to zdjecie nie bylo jeszcze dodane
                if p not in other_images and p not in felix_images and p not in val_images:
                    other_images.append(p)
                    # Sprawdz czy w sasiedztwie jest folder labels
                    lbl_dirs = [p.parent.parent / "labels", p.parent / "labels"]
                    ensure_label_exists_for_image(p, lbl_dirs)

    # Opcjonalne wydzielenie drobnej czesci Felix_data do walidacji (aby val rzetelnie mierzyl jakosc)
    import random
    random.seed(42)
    shuffled_felix = list(felix_images)
    random.shuffle(shuffled_felix)

    n_val_felix = max(1, int(len(shuffled_felix) * val_felix_ratio)) if len(shuffled_felix) > 10 else 0
    val_felix = shuffled_felix[:n_val_felix]
    train_felix = shuffled_felix[n_val_felix:]

    # Usuniecie ewentualnych starych/uszkodzonych plikow *.cache Ultralytics
    for cache_file in concatenated_dir.rglob("*.cache"):
        try:
            cache_file.unlink()
        except Exception:
            pass

    # Przygotowanie manifestu train_weighted.txt z pelnymi sciezkami
    train_lines: List[str] = []

    # Zdjecia GOOD (Felix): powtorzone good_weight razy
    for p in train_felix:
        abs_path = p.resolve().as_posix()
        for _ in range(good_weight):
            train_lines.append(abs_path)

    # Zdjecia OTHER: powtorzone other_weight razy
    for p in other_images:
        abs_path = p.resolve().as_posix()
        for _ in range(other_weight):
            train_lines.append(abs_path)

    # Przygotowanie manifestu val.txt z pelnymi sciezkami
    val_lines: List[str] = []
    for p in val_images:
        val_lines.append(p.resolve().as_posix())

    # Zapis plikow manifestu
    train_txt_path = (concatenated_dir / "train_weighted.txt").resolve()
    val_txt_path = (concatenated_dir / "val.txt").resolve()
    train_txt_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_txt_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    # Zapis data_weighted.yaml z jednoznacznymi, prawidlowymi sciezkami
    yaml_data = {
        "path": concatenated_dir.resolve().as_posix(),
        "train": train_txt_path.as_posix(),
        "val": val_txt_path.as_posix(),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }
    yaml_path = concatenated_dir / "data_weighted.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    # Obliczenie statystyk
    total_good_train = len(train_felix)
    total_other_train = len(other_images)
    weighted_good = total_good_train * good_weight
    weighted_other = total_other_train * other_weight
    total_steps = weighted_good + weighted_other
    pct_good = (weighted_good / total_steps * 100) if total_steps > 0 else 0

    print("\n" + "=" * 75)
    print(" SUKCES! ZBIOR data_concatenated ZOSTAL ZWAGOWANY:")
    print("=" * 75)
    print(f"1. DANE IDEALNE (Felix_data):")
    print(f"   - Unikalne zdjęcia:        {total_good_train} szt.")
    print(f"   - Waga (mnożnik):          {good_weight}x")
    print(f"   - Próbek w epoce:          {weighted_good} kroków")
    print(f"2. DANE POZOSTAŁE (other / synthetic):")
    print(f"   - Unikalne zdjęcia:        {total_other_train} szt.")
    print(f"   - Waga:                    {other_weight}x")
    print(f"   - Próbek w epoce:          {weighted_other} kroków")
    print(f"3. BILANS TRENINGU:")
    print(f"   - Łącznie kroków w epoce:  {total_steps}")
    print(f"   - WPŁYW DANYCH FELIXA:     {pct_good:.1f}% WSZYSTKICH GRADIENTÓW W KAŻDEJ EPOCE!")
    print(f"4. WALIDACJA (val.txt):       {len(val_images)} szt.")
    print(f"5. WYGENEROWANE PLIKI:")
    print(f"   - Manifest treningowy:     {train_txt_path}")
    print(f"   - Manifest walidacyjny:    {val_txt_path}")
    print(f"   - Konfiguracja YOLO:       {yaml_path}")
    print("=" * 75)
    print("\n>>> JAK ODPALIĆ TRENING PRZEZ data_eval.py NA CUDA:")
    print(f"    python data_eval.py --data {yaml_path.as_posix()} --epochs 50 --batch 16 --device 0\n")

    return yaml_path


def process_good_other_folders(
    good_dir: Path,
    other_dir: Path,
    output_dir: Path,
    good_weight: int = 15,
    other_weight: int = 1,
    val_ratio: float = 0.15,
) -> Path:
    """Obsluga prostych folderow good/ i other/."""
    good_dir = Path(good_dir).resolve()
    other_dir = Path(other_dir).resolve()
    output_dir = Path(output_dir).resolve()

    good_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    good_images = [p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
    other_images = [p for p in other_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]

    if not good_images and not other_images:
        print("\n" + "!" * 70)
        print("[!] FOLDERY WEJŚCIOWE SĄ PUSTE!")
        print(f"    - Good:  {good_dir}")
        print(f"    - Other: {other_dir}")
        print("!" * 70 + "\n")
        return output_dir

    print(f"Wagowanie z folderów: good ({len(good_images)} zdj.), other ({len(other_images)} zdj.)")

    out_train_img = output_dir / "images" / "train"
    out_val_img = output_dir / "images" / "val"
    out_train_lbl = output_dir / "labels" / "train"
    out_val_lbl = output_dir / "labels" / "val"
    for d in [out_train_img, out_val_img, out_train_lbl, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)

    def split_set(items, ratio):
        shuffled = list(items)
        random.shuffle(shuffled)
        n = max(1, int(len(shuffled) * ratio)) if len(shuffled) > 3 else 0
        return shuffled[n:], shuffled[:n]

    tr_good, val_good = split_set(good_images, val_ratio)
    tr_other, val_other = split_set(other_images, val_ratio)

    def copy_file(src, dst):
        if not dst.exists():
            shutil.copy2(src, dst)

    search_dirs = [good_dir, other_dir, REPO_ROOT / "data"]
    train_lines = []
    for p in tr_good:
        dest = out_train_img / p.name
        copy_file(p, dest)
        ensure_label_exists_for_image(p, search_dirs)
        lbl = p.parent / f"{p.stem}.txt"
        if lbl.exists():
            copy_file(lbl, out_train_lbl / f"{p.stem}.txt")
        for _ in range(good_weight):
            train_lines.append(f"images/train/{p.name}")

    for p in tr_other:
        dest = out_train_img / p.name
        copy_file(p, dest)
        ensure_label_exists_for_image(p, search_dirs)
        lbl = p.parent / f"{p.stem}.txt"
        if lbl.exists():
            copy_file(lbl, out_train_lbl / f"{p.stem}.txt")
        for _ in range(other_weight):
            train_lines.append(f"images/train/{p.name}")

    val_lines = []
    for p in val_good + val_other:
        dest = out_val_img / p.name
        copy_file(p, dest)
        ensure_label_exists_for_image(p, search_dirs)
        lbl = p.parent / f"{p.stem}.txt"
        if lbl.exists():
            copy_file(lbl, out_val_lbl / f"{p.stem}.txt")
        val_lines.append(f"images/val/{p.name}")

    (output_dir / "train_weighted.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    yaml_data = {
        "path": ".",
        "train": "train_weighted.txt",
        "val": "val.txt",
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"\n[OK] Utworzono zwagowany zbiór w: {output_dir}")
    print(f"Plik data.yaml: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="weight_dataset - Wagowanie danych (Felix_data vs reszta lub good/other) dla YOLOv8 Small."
    )
    parser.add_argument(
        "--concatenated-dir",
        type=str,
        default=None,
        help="Ścieżka do folderu data_concatenated (jeśli istnieje, skrypt automatycznie zwaguje Felix_data vs resztę)",
    )
    parser.add_argument("--good-dir", type=Path, default=None, help="Folder ze zdjęciami good")
    parser.add_argument("--other-dir", type=Path, default=None, help="Folder ze zdjęciami other")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset_weighted", help="Katalog wyjściowy")
    parser.add_argument("--good-weight", type=int, default=15, help="Waga dla zdjęć idealnych (Felix_data / good) - domyślnie 15x")
    parser.add_argument("--other-weight", type=int, default=1, help="Waga dla pozostałych zdjęć - domyślnie 1x")

    args = parser.parse_args()

    # Automatyczne sprawdzenie, czy istnieje data_concatenated
    concat_candidates = [
        args.concatenated_dir,
        REPO_ROOT / "data" / "data_concatenated",
        REPO_ROOT / "data_concatenated",
    ]
    detected_concat = None
    for cand in concat_candidates:
        if cand and Path(cand).exists():
            detected_concat = Path(cand)
            break

    if detected_concat is not None:
        process_concatenated_dataset(
            concatenated_dir=detected_concat,
            good_weight=args.good_weight,
            other_weight=args.other_weight,
        )
    else:
        # Tryb z folderami good/ i other/
        g_dir = args.good_dir or (REPO_ROOT / "good")
        o_dir = args.other_dir or (REPO_ROOT / "other")
        process_good_other_folders(
            good_dir=g_dir,
            other_dir=o_dir,
            output_dir=args.output_dir,
            good_weight=args.good_weight,
            other_weight=args.other_weight,
        )


if __name__ == "__main__":
    main()
