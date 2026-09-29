"""
weight_dataset.py

Skrypt nakladajacy wagi na zdjecia treningowe w zbiorach YOLO / COCO.
Umozliwia zrownowazenie zbioru, w ktorym jest malo zdjec idealnych (wysokiej jakosci,
zrobionych z gory, idealne etykiety) oraz duzo zdjec slabej jakosci.

Jak dziala wazenie probek (Sample Weighting):
  W uczeniu maszynowym (SGD / Adam) zwiekszenie wagi probki W-krotnie jest
  matematycznie rownowazne W-krotnemu zwiekszeniu czestotliwosci jej wystepowania
  w epoce treningowej (Weighted Resampling / Oversampling).
  
  Dzieki temu model przy kazdej epoce widzi zdjecia idealne znacznie czesciej
  (kazdorazowo z losowymi augmentacjami: obrot, skala, barwa, mozaika),
  a gradienty z nich dominuja nad szumem ze zdjec niskiej jakosci.
  Zarazem spelniony jest warunek: trenujemy na WSZYSTKICH dostepnych zdjeciach.

Mozliwosci wskazania zdjec idealnych:
  1. Plik tekstowy z lista nazw:       --ideal-list ideal_images.txt
  2. Wzorzec w nazwie (regex/prefix):  --ideal-pattern "ragwort*"
  3. Osobny folder ze zdjeciami:       --ideal-dir path/to/ideal/
  4. Plik CSV z wagami per plik:       --weights-csv weights.csv
  5. Tryb interaktywny / szablon:      --create-template

Wyjscie:
  - train_weighted.txt  (lista sciezek do zdjec z uwzglednieniem powtorzen wagowych)
  - data_weighted.yaml  (gotowy plik konfiguracji YOLO z wazonym zbiorem)
  - weights_summary.json (statystyki wag i rozkladu)
  - opcjonalnie zmaterializowany folder lub wazony COCO JSON dla DEIM/DINO.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_dataset_images(dataset_dir: Path, split: str = "train") -> List[Path]:
    """Wyszukuje wszystkie obrazy w danym splicie zbioru YOLO."""
    candidates = [
        dataset_dir / split / "images",
        dataset_dir / "images" / split,
        dataset_dir / split,
    ]
    img_dir = None
    for c in candidates:
        if c.exists() and c.is_dir():
            img_dir = c
            break

    if img_dir is None:
        raise FileNotFoundError(f"Nie znaleziono katalogu zdjec dla splitu '{split}' w {dataset_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
    return sorted(images)


def load_ideal_set(
    ideal_list_file: Optional[Path] = None,
    ideal_dir: Optional[Path] = None,
    ideal_pattern: Optional[str] = None,
    weights_csv: Optional[Path] = None,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Zwraca:
      - zbior nazw plikow oznaczonych jako 'idealne'
      - slownik bezposrednich wag {nazwa_pliku: waga} (jesli podano CSV)
    """
    ideal_filenames: Set[str] = set()
    custom_weights: Dict[str, float] = {}

    # 1. Z pliku tekstowego (jedna nazwa w wierszu)
    if ideal_list_file and ideal_list_file.exists():
        with open(ideal_list_file, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip()
                if clean and not clean.startswith("#"):
                    ideal_filenames.add(Path(clean).name)
        print(f"[Wagi] Wczytano {len(ideal_filenames)} idealnych zdjec z: {ideal_list_file}")

    # 2. Z katalogu z idealnymi zdjeciami
    if ideal_dir and ideal_dir.exists():
        for p in ideal_dir.iterdir():
            if p.is_file():
                ideal_filenames.add(p.name)
        print(f"[Wagi] Dodano zdjecia z folderu idealnych: {ideal_dir} (lacznie: {len(ideal_filenames)})")

    # 3. Z pliku CSV (filename, weight)
    if weights_csv and weights_csv.exists():
        with open(weights_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                name = Path(row[0].strip()).name
                try:
                    w = float(row[1].strip())
                    custom_weights[name] = w
                except (IndexError, ValueError):
                    custom_weights[name] = 1.0
        print(f"[Wagi] Wczytano wagi dla {len(custom_weights)} plikow z CSV: {weights_csv}")

    return ideal_filenames, custom_weights


def build_weighted_dataset(
    data_yaml_path: Path,
    ideal_filenames: Set[str],
    custom_weights: Dict[str, float],
    ideal_pattern: Optional[str] = None,
    ideal_weight: int = 10,
    poor_weight: int = 1,
    output_dir: Optional[Path] = None,
    materialize_images: bool = False,
    update_coco_json: Optional[Path] = None,
) -> Path:
    """
    Glowna logika tworzenia wazonego manifestu dla YOLO / DEIM.
    """
    data_yaml_path = Path(data_yaml_path).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Brak pliku konfiguracji: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    # Ustal katalog bazowy zbioru
    base_path_raw = yaml_cfg.get("path", "")
    if base_path_raw:
        base_path = Path(base_path_raw)
        if not base_path.is_absolute():
            base_path = (data_yaml_path.parent / base_path).resolve()
            if not base_path.exists():
                base_path = (REPO_ROOT / base_path_raw).resolve()
    else:
        base_path = data_yaml_path.parent

    print(f"[Wagi] Baza danych: {base_path}")
    train_images = find_dataset_images(base_path, split="train")
    print(f"[Wagi] Znaleziono {len(train_images)} unikalnych zdjec treningowych.")

    if output_dir is None:
        output_dir = base_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obliczenie wag dla kazdego pliku
    weighted_manifest_lines: List[str] = []
    stats = {
        "total_unique": len(train_images),
        "ideal_count": 0,
        "poor_count": 0,
        "total_samples_per_epoch": 0,
        "ideal_effective_share_pct": 0.0,
        "weights_distribution": {},
    }

    per_image_weights: Dict[str, int] = {}

    for img_path in train_images:
        fname = img_path.name
        is_ideal = False

        # Sprawdz bezposrednia wage z CSV
        if fname in custom_weights:
            w = max(1, int(round(custom_weights[fname])))
            is_ideal = (w > poor_weight)
        # Sprawdz liste idealnych
        elif fname in ideal_filenames:
            w = ideal_weight
            is_ideal = True
        # Sprawdz pattern (np. 'ragwort*' lub 'ideal*')
        elif ideal_pattern and fnmatch.fnmatch(fname, ideal_pattern):
            w = ideal_weight
            is_ideal = True
        else:
            w = poor_weight
            is_ideal = False

        per_image_weights[fname] = w
        if is_ideal:
            stats["ideal_count"] += 1
        else:
            stats["poor_count"] += 1

        # Uzyj sciezki ze slaszami (bezpieczne dla YOLO na Windows i Linux)
        clean_path_str = str(img_path.resolve()).replace("\\", "/")

        # Powtorz sciezke W razy w manifeście
        for _ in range(w):
            weighted_manifest_lines.append(clean_path_str)

    stats["total_samples_per_epoch"] = len(weighted_manifest_lines)
    ideal_total_slots = stats["ideal_count"] * ideal_weight
    if stats["total_samples_per_epoch"] > 0:
        stats["ideal_effective_share_pct"] = round(
            (ideal_total_slots / stats["total_samples_per_epoch"]) * 100, 2
        )

    # Zapisz manifest train_weighted.txt
    manifest_path = output_dir / "train_weighted.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(weighted_manifest_lines) + "\n")
    print(f"\n[Wagi] Utworzono manifest wagowy: {manifest_path}")
    print(f"       -> Liczba probek na epoke: {stats['total_samples_per_epoch']} (wczesniej: {stats['total_unique']})")
    print(f"       -> Zdjecia idealne ({stats['ideal_count']} szt.) stanowia teraz {stats['ideal_effective_share_pct']}% gradientow w kazdej epoce!")

    # Zbuduj nowy data_weighted.yaml
    weighted_yaml_cfg = dict(yaml_cfg)
    weighted_yaml_cfg["path"] = str(base_path.resolve()).replace("\\", "/")
    weighted_yaml_cfg["train"] = str(manifest_path.resolve()).replace("\\", "/")

    weighted_yaml_path = output_dir / "data_weighted.yaml"
    with open(weighted_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(weighted_yaml_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[Wagi] Zapisano wazona konfiguracje YOLO: {weighted_yaml_path}")

    # Zapisz raport podsumowujacy JSON
    summary_path = output_dir / "weights_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Opcjonalnie: Materializacja fizyczna (np. dla loaderow niewspierajacych txt)
    if materialize_images:
        mat_img_dir = output_dir / "train_weighted_images"
        mat_lbl_dir = output_dir / "train_weighted_labels"
        mat_img_dir.mkdir(parents=True, exist_ok=True)
        mat_lbl_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Wagi] Materializacja fizycznych kopii/symlinkow do: {mat_img_dir}...")

        idx = 0
        for img_path in train_images:
            fname = img_path.name
            w = per_image_weights[fname]
            stem = img_path.stem
            ext = img_path.suffix

            # Etykieta
            lbl_candidate = img_path.parent.parent / "labels" / f"{stem}.txt"
            if not lbl_candidate.exists():
                lbl_candidate = img_path.with_suffix(".txt")

            for rep in range(w):
                target_img_name = f"{stem}_w{rep}{ext}"
                target_lbl_name = f"{stem}_w{rep}.txt"
                
                # Proba stworzenia hardlinku (oszczedza 100% dysku), w razie bledu kopiowanie
                target_img = mat_img_dir / target_img_name
                if not target_img.exists():
                    try:
                        os.link(img_path, target_img)
                    except Exception:
                        shutil.copy2(img_path, target_img)

                if lbl_candidate.exists():
                    target_lbl = mat_lbl_dir / target_lbl_name
                    if not target_lbl.exists():
                        try:
                            os.link(lbl_candidate, target_lbl)
                        except Exception:
                            shutil.copy2(lbl_candidate, target_lbl)
                idx += 1
        print(f"[Wagi] Zmaterializowano {idx} plikow obrazow i etykiet.")

    # Opcjonalnie: Wazenie COCO JSON (dla DEIM/DINO)
    if update_coco_json and Path(update_coco_json).exists():
        update_coco_annotations(Path(update_coco_json), per_image_weights, output_dir)

    return weighted_yaml_path


def update_coco_annotations(
    coco_json_path: Path,
    weights_map: Dict[str, int],
    output_dir: Path,
) -> Path:
    """Duplikuje wpisy w pliku COCO instances_train.json wg podanych wag."""
    print(f"[COCO] Aktualizacja wag w COCO JSON: {coco_json_path}")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # Mapowanie image_id -> lista adnotacji
    img_to_anns: Dict[int, List[dict]] = {}
    for ann in annotations:
        iid = ann["image_id"]
        img_to_anns.setdefault(iid, []).append(ann)

    new_images = []
    new_annotations = []
    next_img_id = max((img["id"] for img in images), default=0) + 1
    next_ann_id = max((ann["id"] for ann in annotations), default=0) + 1

    for img in images:
        fname = Path(img["file_name"]).name
        w = weights_map.get(fname, 1)

        # Oryginalny wpis
        new_images.append(img)
        for ann in img_to_anns.get(img["id"], []):
            new_annotations.append(ann)

        # Dodatkowe powtorzenia wg wagi
        for rep in range(1, w):
            cloned_img = dict(img)
            cloned_img["id"] = next_img_id
            new_images.append(cloned_img)

            for ann in img_to_anns.get(img["id"], []):
                cloned_ann = dict(ann)
                cloned_ann["id"] = next_ann_id
                cloned_ann["image_id"] = next_img_id
                new_annotations.append(cloned_ann)
                next_ann_id += 1

            next_img_id += 1

    weighted_coco = dict(data)
    weighted_coco["images"] = new_images
    weighted_coco["annotations"] = new_annotations

    out_coco_path = output_dir / "instances_train_weighted.json"
    with open(out_coco_path, "w", encoding="utf-8") as f:
        json.dump(weighted_coco, f)

    print(f"[COCO] Zapisano wazony plik COCO: {out_coco_path} (obrazow: {len(new_images)})")
    return out_coco_path


def create_template_file(data_yaml_path: Path, output_file: Path = Path("ideal_images_template.txt")):
    """Generuje plik szablonu z wszystkimi zdjeciami, ulatwiajac uzytkownikowi zaznaczenie idealnych."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    base_path_raw = yaml_cfg.get("path", "")
    base_path = Path(base_path_raw)
    if not base_path.is_absolute():
        base_path = (data_yaml_path.parent / base_path).resolve()
        if not base_path.exists():
            base_path = (REPO_ROOT / base_path_raw).resolve()

    train_images = find_dataset_images(base_path, split="train")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# SZABLON WYBORU ZDJEC IDEALNYCH\n")
        f.write("# Zostaw w tym pliku tylko te zdjecia, ktore sa 'idealne' (np. ladny widok z gory),\n")
        f.write("# albo dopisz wage po przecinku: nazwa.jpg, 15\n")
        f.write("# Linie zaczynajace sie od # sa ignorowane.\n\n")
        for img in train_images:
            f.write(f"# {img.name}\n")

    print(f"[Szablon] Wygenerowano szablon z {len(train_images)} zdjeciami w: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Naklada wagi na zdjecia treningowe w zbiorze YOLO/COCO (Weighted Resampling)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "combined_dataset" / "data.yaml",
        help="Sciezka do data.yaml (np. data/combined_dataset/data.yaml lub data/ragwort_segmentation.yaml)",
    )
    parser.add_argument(
        "--ideal-list",
        type=Path,
        help="Sciezka do pliku .txt z lista nazw idealnych zdjec (jedno na wiersz)",
    )
    parser.add_argument(
        "--ideal-dir",
        type=Path,
        help="Katalog zawierajacy idealne zdjecia",
    )
    parser.add_argument(
        "--ideal-pattern",
        type=str,
        default=None,
        help="Wzorzec nazwy dla idealnych zdjec (np. 'ragwort*' lub '*topdown*')",
    )
    parser.add_argument(
        "--weights-csv",
        type=Path,
        help="Plik CSV z wagami per zdjecie (format: filename, weight)",
    )
    parser.add_argument(
        "--ideal-weight",
        type=int,
        default=10,
        help="Mnoznik wagi dla zdjec idealnych (domyslnie: 10x czesciej w epoce)",
    )
    parser.add_argument(
        "--poor-weight",
        type=int,
        default=1,
        help="Waga dla pozostalych (slabych) zdjec (domyslnie: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Gdzie zapisac train_weighted.txt i data_weighted.yaml (domyslnie obok data.yaml)",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Tworzy fizyczny folder z kopiami/hardlinkami dla frameworkow wymagajacych folderu",
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        help="Opcjonalna sciezka do instances_train.json dla DEIM/DINO (tworzy instances_train_weighted.json)",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Tworzy plik ideal_images_template.txt z lista wszystkich plikow do uzupelnienia",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.create_template:
        out_tmpl = Path("ideal_images_template.txt")
        create_template_file(args.data, out_tmpl)
        print(f"\nOtworz plik '{out_tmpl}', odkomentuj idealne zdjecia i uruchom:")
        print(f"python src/scripts/weight_dataset.py --data {args.data} --ideal-list {out_tmpl} --ideal-weight 10")
        return

    # Wczytanie informacji o zdjeciach idealnych
    ideal_filenames, custom_weights = load_ideal_set(
        ideal_list_file=args.ideal_list,
        ideal_dir=args.ideal_dir,
        ideal_pattern=args.ideal_pattern,
        weights_csv=args.weights_csv,
    )

    # Domyslny fallback: jesli uzytkownik nic nie podal, sprawdz czy sa zdjecia z prefiksem 'ragwort*'
    pattern = args.ideal_pattern
    if not ideal_filenames and not custom_weights and not pattern:
        print("[Info] Nie podano --ideal-list ani --ideal-pattern.")
        print("       Sprawdzam czy wystepuja pliki z wzorcem 'ragwort*'...")
        pattern = "ragwort*"

    weighted_yaml = build_weighted_dataset(
        data_yaml_path=args.data,
        ideal_filenames=ideal_filenames,
        custom_weights=custom_weights,
        ideal_pattern=pattern,
        ideal_weight=args.ideal_weight,
        poor_weight=args.poor_weight,
        output_dir=args.output_dir,
        materialize_images=args.materialize,
        update_coco_json=args.coco_json,
    )

    print("\n" + "=" * 70)
    print(" GOTOWE! JAK TRENOWAC MODEL Z WAGAMI:")
    print("=" * 70)
    print(f"1. Standardowe YOLOv8 CLI:")
    print(f"   yolo detect train data=\"{weighted_yaml}\" epochs=30 imgsz=640 batch=8")
    print(f"\n2. W Twoim skrypcie Python / segmentacji:")
    print(f"   python src/train_evaluation/train_ragwort_segmentation.py --data \"{weighted_yaml}\" --epochs 30")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
