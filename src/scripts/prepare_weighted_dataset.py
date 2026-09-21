"""
prepare_weighted_dataset.py

Skrypt do obslugi struktury:
  - good/   -> zdjecia idealne (wysokiej jakosci, zrobione z gory, precyzyjne)
  - other/  -> pozostale zdjecia (gorszej jakosci, szum, trudne warunki)

Skrypt:
1. Zbiera zdjecia z folderow 'good' i 'other'.
2. Automatycznie dopasowuje etykiety .txt (szuka ich w folderach wejsciowych lub w bazie data/).
3. Przypisuje wagi (np. 15x dla 'good', 1x dla 'other') za pomoca Weighted Resampling.
4. Tworzy w pelni PRZENOSNY zbior danych w folderze 'dataset_export_ready/':
   - relatywne sciezki (dziala na dowolnym systemie: Windows / Linux / Google Colab),
   - plik data.yaml z podpietym train_weighted.txt,
   - gotowe skrypty startowe train.py, run_train.bat i run_train.sh.
5. Opcjonalnie pakuje calosc do 'dataset_export_ready.zip', gotowego do wyslania na pendrive/dysk.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_label_file(img_name: str, search_dirs: List[Path]) -> Optional[Path]:
    """Wyszukuje plik etykiety .txt dla zadanego zdjecia w liscie katalogow."""
    stem = Path(img_name).stem
    txt_name = f"{stem}.txt"

    for d in search_dirs:
        if not d.exists():
            continue
        # Bezposrednio w katalogu
        candidate = d / txt_name
        if candidate.is_file():
            return candidate
        # W podkatalogu labels
        sub_candidate = d / "labels" / txt_name
        if sub_candidate.is_file():
            return sub_candidate
        # Rekurencyjnie w labels (np. train/labels)
        for found in d.glob(f"**/{txt_name}"):
            if found.is_file():
                return found

    return None


def prepare_dataset(
    good_dir: Path,
    other_dir: Path,
    output_dir: Path,
    good_weight: int = 15,
    other_weight: int = 1,
    val_ratio: float = 0.15,
    class_names: Optional[List[str]] = None,
    make_zip: bool = True,
) -> Path:
    if class_names is None:
        class_names = ["ragwort", "objects"]

    good_dir = Path(good_dir).resolve()
    other_dir = Path(other_dir).resolve()
    output_dir = Path(output_dir).resolve()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Utworz foldery wejsciowe jesli nie istnieja
    good_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    good_images = [p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    other_images = [p for p in other_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not good_images and not other_images:
        print(f"\n[!] Foldery sa puste! Utworzono dla Ciebie katalogi:")
        print(f"    - GOOD (zdjecia idealne):  {good_dir}")
        print(f"    - OTHER (pozostale):       {other_dir}")
        print("Wrzuc tam pliki zdjec (i opcjonalnie pliki .txt z etykietami) i uruchom skrypt ponownie.\n")
        return output_dir

    print("=" * 70)
    print(" PRZYGOTOWYWANIE WAZONEGO ZBIORU DANYCH DO EKSPORTU")
    print("=" * 70)
    print(f"Folder GOOD:    {good_dir} ({len(good_images)} zdjec, waga = {good_weight}x)")
    print(f"Folder OTHER:   {other_dir} ({len(other_images)} zdjec, waga = {other_weight}x)")
    print(f"Katalog wyjscia: {output_dir}")
    print("=" * 70)

    # Sciezki do przeszukania w poszukiwaniu istniejacych etykiet .txt
    label_search_dirs = [
        good_dir,
        other_dir,
        REPO_ROOT / "data" / "combined_dataset" / "train" / "labels",
        REPO_ROOT / "data" / "combined_dataset" / "test" / "labels",
        REPO_ROOT / "data" / "ragwort_segmentation" / "train" / "labels",
        REPO_ROOT / "data" / "ragwort_segmentation" / "val" / "labels",
        REPO_ROOT / "data",
    ]

    # Podzial na train i val (zrownowazony dla obu grup)
    import random
    random.seed(42)

    def split_set(items: List[Path], ratio: float) -> Tuple[List[Path], List[Path]]:
        shuffled = list(items)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * ratio)) if len(shuffled) > 3 else 0
        return shuffled[n_val:], shuffled[:n_val]

    train_good, val_good = split_set(good_images, val_ratio)
    train_other, val_other = split_set(other_images, val_ratio)

    train_all_images = [(p, "good", good_weight) for p in train_good] + [
        (p, "other", other_weight) for p in train_other
    ]
    val_all_images = [(p, "good") for p in val_good] + [(p, "other") for p in val_other]

    # Przygotowanie struktury docelowej
    out_train_img = output_dir / "images" / "train"
    out_val_img = output_dir / "images" / "val"
    out_train_lbl = output_dir / "labels" / "train"
    out_val_lbl = output_dir / "labels" / "val"

    for d in [out_train_img, out_val_img, out_train_lbl, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_file_or_link(src: Path, dst: Path):
        if not dst.exists():
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    # Kopiowanie zdjec treningowych i etykiet
    missing_labels = []
    train_txt_relative_lines: List[str] = []

    for img_path, category, weight in train_all_images:
        dest_img = out_train_img / img_path.name
        copy_file_or_link(img_path, dest_img)

        # Dopasowanie etykiety
        lbl_file = find_label_file(img_path.name, label_search_dirs)
        dest_lbl = out_train_lbl / f"{img_path.stem}.txt"
        if lbl_file and lbl_file.exists():
            copy_file_or_link(lbl_file, dest_lbl)
        else:
            # Tworzymy pusta etykiete (obraz tla)
            dest_lbl.write_text("", encoding="utf-8")
            missing_labels.append(img_path.name)

        # Relatywna sciezka dla YOLO (uzywamy forward slash dla przenosnosci)
        rel_img_path = f"images/train/{img_path.name}"
        for _ in range(weight):
            train_txt_relative_lines.append(rel_img_path)

    # Kopiowanie zdjec walidacyjnych
    val_txt_relative_lines: List[str] = []
    for img_path, category in val_all_images:
        dest_img = out_val_img / img_path.name
        copy_file_or_link(img_path, dest_img)

        lbl_file = find_label_file(img_path.name, label_search_dirs)
        dest_lbl = out_val_lbl / f"{img_path.stem}.txt"
        if lbl_file and lbl_file.exists():
            copy_file_or_link(lbl_file, dest_lbl)
        else:
            dest_lbl.write_text("", encoding="utf-8")

        val_txt_relative_lines.append(f"images/val/{img_path.name}")

    if missing_labels:
        print(f"[Uwaga] Dla {len(missing_labels)} zdjec nie znaleziono pliku .txt (zapisano jako puste tlo).")

    # Zapisz train_weighted.txt i val.txt
    manifest_train = output_dir / "train_weighted.txt"
    with open(manifest_train, "w", encoding="utf-8") as f:
        f.write("\n".join(train_txt_relative_lines) + "\n")

    manifest_val = output_dir / "val.txt"
    with open(manifest_val, "w", encoding="utf-8") as f:
        f.write("\n".join(val_txt_relative_lines) + "\n")

    # Zapisz data.yaml z RELATYWNYMI sciezkami (przenosny na dowolny komputer!)
    data_yaml_content = {
        "path": ".",  # Kluczowe dla przenosnosci: biezacy katalog z data.yaml
        "train": "train_weighted.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, allow_unicode=True)

    # Generowanie gotowego skryptu treningowego train.py dla drugiego komputera
    train_script_content = f'''"""
train.py - Gotowy skrypt do odpalenia treningu YOLOv8 na Twoim drugim komputerze.
Uruchomienie:
    python train.py
Lub z wiersza polecen (CLI):
    yolo detect train model=yolov8m.pt data=data.yaml epochs=50 imgsz=640 batch=16 device=0
"""

import torch
from ultralytics import YOLO

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(" START TRENINGU YOLOV8 NA DRUGIM KOMPUTERZE")
    print(f" Urzadzenie: {{device}} (CUDA dostepne: {{torch.cuda.is_available()}})")
    print("=" * 60)

    # Modele YOLOv8 wieksze od nano:
    # - 'yolov8s.pt' (Small - lekki, ale 3x dokladniejszy od nano)
    # - 'yolov8m.pt' (Medium - REKOMENDOWANY kompromis dokladnosci i predkosci)
    # - 'yolov8l.pt' (Large - bardzo dokladny, wymaga min. 8-12 GB VRAM)
    # - 'yolov8x.pt' (XLarge - maksymalna dokladnosc, wymaga mocnej karty)
    
    model_name = "yolov8s.pt"
    print(f"Pobieranie i ladowanie modelu: {{model_name}}...")
    model = YOLO(model_name)

    # Uruchomienie treningu z uzyciem zbalansowanego pliku data.yaml
    results = model.train(
        data="data.yaml",
        epochs=50,          # 50 epok da swietny rezultat na zwagowanych danych
        imgsz=640,          # rozdzielczosc
        batch=16,           # zmniejsz do 8 jesli zabraknie pamieci VRAM
        device=device,
        workers=4,
        save=True,
        project="runs_ragwort",
        name="yolov8s_weighted",
        exist_ok=True,
    )

    print("\\nTrening zakonczony!")
    print("Najlepsze wagi zapisuja sie w: runs_ragwort/yolov8s_weighted/weights/best.pt")

if __name__ == "__main__":
    main()
'''
    with open(output_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(train_script_content)

    # Skrypt startowy .bat (dla Windows na drugim PC)
    bat_content = "@echo off\necho Instalowanie zaleznosci...\npip install ultralytics torch torchvision\necho Uruchamianie treningu YOLOv8 Medium...\npython train.py\npause\n"
    with open(output_dir / "run_train.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)

    # Skrypt startowy .sh (dla Linux / Google Colab)
    sh_content = "#!/bin/bash\npip install ultralytics torch torchvision\npython train.py\n"
    with open(output_dir / "run_train.sh", "w", encoding="utf-8") as f:
        f.write(sh_content)

    total_good_train = len(train_good)
    total_other_train = len(train_other)
    weighted_good_samples = total_good_train * good_weight
    weighted_other_samples = total_other_train * other_weight
    total_samples = weighted_good_samples + weighted_other_samples

    pct_good = (weighted_good_samples / total_samples * 100) if total_samples > 0 else 0

    print("\n" + "=" * 70)
    print(" SUKCES! ZBIOR ZOSTAL SKONFIGUROWANY I PRZYGOTOWANY")
    print("=" * 70)
    print(f"Trening (train):")
    print(f"  - Unikalne zdjecia GOOD:  {total_good_train} szt. (waga {good_weight}x -> {weighted_good_samples} probek)")
    print(f"  - Unikalne zdjecia OTHER: {total_other_train} szt. (waga {other_weight}x -> {weighted_other_samples} probek)")
    print(f"  - Lacznie na epoke:       {total_samples} krokow treningowych")
    print(f"  - Udzial zdjec GOOD:      {pct_good:.1f}% gradientow w kazdej epoce!")
    print(f"Walidacja (val):")
    print(f"  - Zdjecia testowe:        {len(val_all_images)} szt.")
    print("=" * 70)

    # Opcjonalne spakowanie do ZIP dla latwego transferu
    if make_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        print(f"\n[Pakowanie] Tworzenie archiwum ZIP do transferu: {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_p = Path(root) / file
                    rel_p = file_p.relative_to(output_dir.parent)
                    zf.write(file_p, arcname=str(rel_p))
        print(f"[Pakowanie] Gotowe! Archiwum ZIP: {zip_path}")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Przygotowuje wazony, samowystarczalny zbior danych z folderow 'good' i 'other'."
    )
    parser.add_argument(
        "--good-dir",
        type=Path,
        default=REPO_ROOT / "dataset_input" / "good",
        help="Folder ze zdjeciami idealnymi (domyslnie: dataset_input/good)",
    )
    parser.add_argument(
        "--other-dir",
        type=Path,
        default=REPO_ROOT / "dataset_input" / "other",
        help="Folder z pozostalymi zdjeciami (domyslnie: dataset_input/other)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dataset_export_ready",
        help="Katalog docelowy przygotowany do transferu",
    )
    parser.add_argument(
        "--good-weight",
        type=int,
        default=15,
        help="Waga / mnoznik dla zdjec idealnych (domyslnie: 15x)",
    )
    parser.add_argument(
        "--other-weight",
        type=int,
        default=1,
        help="Waga dla pozostalych zdjec (domyslnie: 1x)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Nie twórz pliku ZIP",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Sprawdz czy uzytkownik nie stworzyl folderow 'good' i 'other' bezposrednio w root projektu
    good_dir = args.good_dir
    other_dir = args.other_dir

    if (REPO_ROOT / "good").exists() and any((REPO_ROOT / "good").iterdir()):
        good_dir = REPO_ROOT / "good"
    if (REPO_ROOT / "other").exists() and any((REPO_ROOT / "other").iterdir()):
        other_dir = REPO_ROOT / "other"

    prepare_dataset(
        good_dir=good_dir,
        other_dir=other_dir,
        output_dir=args.output_dir,
        good_weight=args.good_weight,
        other_weight=args.other_weight,
        make_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()
