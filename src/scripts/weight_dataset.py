"""
weight_dataset.py - Skrypt do automatycznego wagowania danych dla RagwortDetection.

Zgodnie ze standardem projektu, wszystkie wygenerowane zbiory i manifesty
zapisywane są w uporządkowanym katalogu `outputs/datasets/`:
  - outputs/datasets/data_weighted/data_weighted.yaml  (dla YOLO)
  - outputs/datasets/data_weighted/train_weighted.txt (manifest YOLO)
  - outputs/datasets/data_weighted/val.txt            (manifest walidacyjny YOLO)
  - outputs/datasets/coco/train/annotations.json      (dla DEIMv2 / DINOv3)
  - outputs/datasets/coco/val/annotations.json        (dla DEIMv2 / DINOv3)
  - outputs/datasets/coco/test/annotations.json       (dla DEIMv2 / DINOv3)

Użycie:
  python src/scripts/weight_dataset.py
  python src/scripts/weight_dataset.py --good-weight 15 --other-weight 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_CONCAT_DIR = REPO_ROOT / "data" / "data_concatenated"
DEFAULT_DATASETS_OUT = REPO_ROOT / "outputs" / "datasets"


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

            xc = ((xmin + xmax) / 2.0) / width
            yc = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        output_txt.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        output_txt.write_text("", encoding="utf-8")
        return False


def sanitize_yolo_label_file(lbl_path: Path):
    """Konwertuje ewentualne wiersze z poligonami (>5 kolumn) na standardowe ramki detekcji (cls xc yc w h)."""
    if not lbl_path.is_file() or lbl_path.stat().st_size == 0:
        return
    try:
        lines = lbl_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 5:
                new_lines.append(" ".join(parts))
            elif len(parts) > 5:
                modified = True
                cls_id = parts[0]
                try:
                    coords = [float(v) for v in parts[1:]]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    if xs and ys:
                        xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
                        ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
                        xc = (xmin + xmax) / 2.0
                        yc = (ymin + ymax) / 2.0
                        w = max(0.0, xmax - xmin)
                        h = max(0.0, ymax - ymin)
                        new_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                except Exception:
                    continue
        if modified:
            lbl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    except Exception:
        pass


def get_expected_label_path(img_path: Path) -> Path:
    """Zwraca ścieżkę pliku etykiety, jakiej oczekuje YOLO (img2label_paths)."""
    posix_path = img_path.resolve().as_posix()
    sa, sb = "/images/", "/labels/"
    if sa in posix_path:
        lbl_str = sb.join(posix_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
        return Path(lbl_str)
    else:
        return img_path.parent / f"{img_path.stem}.txt"


def ensure_label_exists_for_image(img_path: Path, possible_label_dirs: List[Path]) -> Path:
    """Zapewnia, że dla danego zdjęcia istnieje plik etykiety .txt."""
    target_lbl = get_expected_label_path(img_path)
    stem = img_path.stem

    if target_lbl.is_file():
        if target_lbl.stat().st_size > 0:
            sanitize_yolo_label_file(target_lbl)
        return target_lbl

    neighbor_txt = img_path.parent / f"{stem}.txt"
    if neighbor_txt != target_lbl and neighbor_txt.is_file() and neighbor_txt.stat().st_size > 0:
        target_lbl.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(neighbor_txt, target_lbl)
            sanitize_yolo_label_file(target_lbl)
            return target_lbl
        except Exception:
            return neighbor_txt

    for d in possible_label_dirs:
        if not d.exists():
            continue

        txt_cand = d / f"{stem}.txt"
        if txt_cand.is_file() and txt_cand.stat().st_size > 0:
            target_lbl.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(txt_cand, target_lbl)
                sanitize_yolo_label_file(target_lbl)
                return target_lbl
            except Exception:
                return txt_cand

        xml_cand = d / f"{stem}.xml"
        if xml_cand.is_file() and xml_cand.stat().st_size > 0:
            target_lbl.parent.mkdir(parents=True, exist_ok=True)
            if convert_voc_xml_to_yolo_txt(xml_cand, target_lbl):
                return target_lbl

    target_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not target_lbl.is_file():
        target_lbl.write_text("", encoding="utf-8")
    return target_lbl


def export_coco_weighted_dataset(
    concatenated_dir: Path,
    train_felix: List[Path],
    other_images: List[Path],
    val_images: List[Path],
    good_weight: int = 15,
    other_weight: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Eksportuje zwagowany zbiór danych do formatu COCO Detection JSON (outputs/datasets/coco/).
    Wartości file_name są relatywne do concatenated_dir (brak zbędnego powielania plików na dysku).
    """
    concatenated_dir = Path(concatenated_dir).resolve()
    coco_root = output_dir or (DEFAULT_DATASETS_OUT / "coco")
    (coco_root / "train").mkdir(parents=True, exist_ok=True)
    (coco_root / "val").mkdir(parents=True, exist_ok=True)
    (coco_root / "test").mkdir(parents=True, exist_ok=True)

    categories = [{"id": 1, "name": "ragwort", "supercategory": "none"}]

    def parse_img(img_path: Path) -> Tuple[int, int, List[List[float]]]:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 640, 640

        lbl_path = get_expected_label_path(img_path)
        boxes: List[List[float]] = []
        if lbl_path.is_file() and lbl_path.stat().st_size > 0:
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        if cls_id == 0:  # ragwort
                            xc, yc, bw, bh = map(float, parts[1:5])
                            xmin = max(0.0, (xc - bw / 2.0) * w)
                            ymin = max(0.0, (yc - bh / 2.0) * h)
                            box_w = min(float(w) - xmin, bw * w)
                            box_h = min(float(h) - ymin, bh * h)
                            if box_w >= 1.0 and box_h >= 1.0:
                                boxes.append([round(xmin, 2), round(ymin, 2), round(box_w, 2), round(box_h, 2)])
                    except Exception:
                        continue
        return w, h, boxes

    parsed_cache: Dict[Path, Tuple[int, int, List[List[float]]]] = {}

    def get_info(p: Path) -> Tuple[int, int, List[List[float]]]:
        if p not in parsed_cache:
            parsed_cache[p] = parse_img(p)
        return parsed_cache[p]

    print("\n[COCO] Generowanie adnotacji COCO w outputs/datasets/coco/...")

    # 1. Trening ze zwagowanymi danymi
    train_coco = {"images": [], "annotations": [], "categories": categories}
    train_img_id = 1
    train_ann_id = 1

    for p in train_felix:
        w, h, boxes = get_info(p)
        rel_path = p.relative_to(concatenated_dir).as_posix()
        for _ in range(good_weight):
            train_coco["images"].append({
                "id": train_img_id,
                "file_name": rel_path,
                "width": w,
                "height": h,
            })
            for box in boxes:
                train_coco["annotations"].append({
                    "id": train_ann_id,
                    "image_id": train_img_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": round(box[2] * box[3], 2),
                    "iscrowd": 0,
                })
                train_ann_id += 1
            train_img_id += 1

    for p in other_images:
        w, h, boxes = get_info(p)
        rel_path = p.relative_to(concatenated_dir).as_posix()
        for _ in range(other_weight):
            train_coco["images"].append({
                "id": train_img_id,
                "file_name": rel_path,
                "width": w,
                "height": h,
            })
            for box in boxes:
                train_coco["annotations"].append({
                    "id": train_ann_id,
                    "image_id": train_img_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": round(box[2] * box[3], 2),
                    "iscrowd": 0,
                })
                train_ann_id += 1
            train_img_id += 1

    # 2. Walidacja (bez powtórzeń)
    val_coco = {"images": [], "annotations": [], "categories": categories}
    val_img_id = 1
    val_ann_id = 1
    for p in val_images:
        w, h, boxes = get_info(p)
        rel_path = p.relative_to(concatenated_dir).as_posix()
        val_coco["images"].append({
            "id": val_img_id,
            "file_name": rel_path,
            "width": w,
            "height": h,
        })
        for box in boxes:
            val_coco["annotations"].append({
                "id": val_ann_id,
                "image_id": val_img_id,
                "category_id": 1,
                "bbox": box,
                "area": round(box[2] * box[3], 2),
                "iscrowd": 0,
            })
            val_ann_id += 1
        val_img_id += 1

    train_json_path = coco_root / "train" / "annotations.json"
    val_json_path = coco_root / "val" / "annotations.json"
    test_json_path = coco_root / "test" / "annotations.json"

    train_json_path.write_text(json.dumps(train_coco, indent=2), encoding="utf-8")
    val_json_path.write_text(json.dumps(val_coco, indent=2), encoding="utf-8")
    test_json_path.write_text(json.dumps(val_coco, indent=2), encoding="utf-8")

    # Symlinki do zdjęć w outputs/datasets/coco/{split}/images
    for split_name in ["train", "val", "test"]:
        images_link = coco_root / split_name / "images"
        if not images_link.exists() and not images_link.is_symlink():
            try:
                images_link.symlink_to(Path("../../../../data/data_concatenated"))
            except Exception:
                pass

    # Zapis w outputs/combined_dataset_coco dla kompatybilności wstecznej z configiem DEIM
    legacy_root = REPO_ROOT / "outputs" / "combined_dataset_coco"
    for split_name, json_data in [("train", train_coco), ("val", val_coco), ("test", val_coco)]:
        split_dir = legacy_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "annotations.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        link = split_dir / "images"
        if not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(Path("../../../data/data_concatenated"))
            except Exception:
                pass

    print(f"[COCO] Trening COCO: {len(train_coco['images'])} próbek, {len(train_coco['annotations'])} ramek")
    print(f"[COCO] Walidacja COCO: {len(val_coco['images'])} próbek, {len(val_coco['annotations'])} ramek")
    print(f"[COCO] Zapisano w: {coco_root}")

    return {
        "train": train_json_path,
        "val": val_json_path,
        "test": test_json_path,
        "coco_root": coco_root,
    }


def process_concatenated_dataset(
    concatenated_dir: Path,
    good_weight: int = 15,
    other_weight: int = 1,
    val_felix_ratio: float = 0.10,
    export_coco: bool = True,
    output_datasets_dir: Optional[Path] = None,
) -> Path:
    """
    Wagowanie zbioru data_concatenated i zapis do outputs/datasets/.
    """
    concatenated_dir = Path(concatenated_dir).resolve()
    out_dir = output_datasets_dir or DEFAULT_DATASETS_OUT
    yolo_dir = out_dir / "data_weighted"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" WAGOWANIE ZBIORU DANYCH (data_concatenated)")
    print(f" Katalog wejściowy: {concatenated_dir}")
    print(f" Katalog wyjściowy: {out_dir}")
    print("=" * 75)

    felix_images: List[Path] = []
    other_images: List[Path] = []
    val_images: List[Path] = []

    # 1. Felix_data (GOOD)
    felix_dir = concatenated_dir / "Felix_data"
    felix_label_dirs = []
    if felix_dir.exists():
        for d in felix_dir.rglob("*"):
            if d.is_dir() and any(k in d.name.lower() for k in ["annot", "label"]):
                felix_label_dirs.append(d)

        for p in felix_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                felix_images.append(p)
                ensure_label_exists_for_image(p, felix_label_dirs)

    # 2. Walidacja / test
    test_dir = concatenated_dir / "combined_dataset" / "test" / "images"
    test_label_dirs = [concatenated_dir / "combined_dataset" / "test" / "labels"]
    if test_dir.exists():
        for p in test_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                val_images.append(p)
                ensure_label_exists_for_image(p, test_label_dirs)

    # 3. Pozostałe (OTHER)
    comb_train = concatenated_dir / "combined_dataset" / "train" / "images"
    comb_train_lbls = [concatenated_dir / "combined_dataset" / "train" / "labels"]
    if comb_train.exists():
        for p in comb_train.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                other_images.append(p)
                ensure_label_exists_for_image(p, comb_train_lbls)

    synth_dir = concatenated_dir / "synthetic" / "images"
    synth_lbls = [concatenated_dir / "synthetic" / "labels"]
    if synth_dir.exists():
        for p in synth_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                other_images.append(p)
                ensure_label_exists_for_image(p, synth_lbls)

    split_dir = concatenated_dir / "synthetic_dataset_split"
    if split_dir.exists():
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
                if p not in other_images and p not in felix_images and p not in val_images:
                    other_images.append(p)
                    lbl_dirs = [p.parent.parent / "labels", p.parent / "labels"]
                    ensure_label_exists_for_image(p, lbl_dirs)

    # Wydzielenie próbki Felix_data do walidacji
    import random
    random.seed(42)
    shuffled_felix = list(felix_images)
    random.shuffle(shuffled_felix)

    n_val_felix = max(1, int(len(shuffled_felix) * val_felix_ratio)) if len(shuffled_felix) > 10 else 0
    val_felix = shuffled_felix[:n_val_felix]
    train_felix = shuffled_felix[n_val_felix:]
    val_images.extend(val_felix)

    # Czyszczenie starych plików cache
    for cache_file in concatenated_dir.rglob("*.cache"):
        try:
            cache_file.unlink()
        except Exception:
            pass

    # Przygotowanie manifestów YOLO (ze ścieżkami bezwzględnymi lub relatywnymi)
    train_lines: List[str] = []
    for p in train_felix:
        rel = p.resolve().as_posix()
        for _ in range(good_weight):
            train_lines.append(rel)

    for p in other_images:
        rel = p.resolve().as_posix()
        for _ in range(other_weight):
            train_lines.append(rel)

    val_lines: List[str] = [p.resolve().as_posix() for p in val_images]

    train_txt_path = (yolo_dir / "train_weighted.txt").resolve()
    val_txt_path = (yolo_dir / "val.txt").resolve()
    train_txt_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_txt_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    yaml_data = {
        "train": str(train_txt_path),
        "val": str(val_txt_path),
        "test": str(val_txt_path),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }
    yaml_path = yolo_dir / "data_weighted.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    # Kopia do data/data_concatenated/ dla wygody
    try:
        shutil.copy2(yaml_path, concatenated_dir / "data_weighted.yaml")
        shutil.copy2(train_txt_path, concatenated_dir / "train_weighted.txt")
        shutil.copy2(val_txt_path, concatenated_dir / "val.txt")
    except Exception:
        pass

    # Eksport do COCO
    coco_paths = {}
    if export_coco:
        coco_paths = export_coco_weighted_dataset(
            concatenated_dir=concatenated_dir,
            train_felix=train_felix,
            other_images=other_images,
            val_images=val_images,
            good_weight=good_weight,
            other_weight=other_weight,
            output_dir=out_dir / "coco",
        )

    # Statystyki
    total_good_train = len(train_felix)
    total_other_train = len(other_images)
    weighted_good = total_good_train * good_weight
    weighted_other = total_other_train * other_weight
    total_steps = weighted_good + weighted_other
    pct_good = (weighted_good / total_steps * 100) if total_steps > 0 else 0

    print("\n" + "=" * 75)
    print(" SUKCES! ZBIÓR ZOSTAŁ ZWAGOWANY:")
    print("=" * 75)
    print("1. DANE IDEALNE (Felix_data):")
    print(f"   - Unikalne zdjęcia:        {total_good_train} szt.")
    print(f"   - Waga (mnożnik):          {good_weight}x")
    print(f"   - Próbek w epoce:          {weighted_good} kroków")
    print("2. DANE POZOSTAŁE (other / synthetic):")
    print(f"   - Unikalne zdjęcia:        {total_other_train} szt.")
    print(f"   - Waga:                    {other_weight}x")
    print(f"   - Próbek w epoce:          {weighted_other} kroków")
    print("3. BILANS TRENINGU:")
    print(f"   - Łącznie kroków w epoce:  {total_steps}")
    print(f"   - WPŁYW DANYCH FELIXA:     {pct_good:.1f}% WSZYSTKICH GRADIENTÓW W KAŻDEJ EPOCE!")
    print(f"4. WALIDACJA (val.txt):       {len(val_images)} szt.")
    print("5. ZAPISANE STRUKTURY W outputs/datasets/:")
    print(f"   - Konfiguracja YOLO:       {yaml_path}")
    print(f"   - Manifest treningowy:     {train_txt_path}")
    print(f"   - Manifest walidacyjny:    {val_txt_path}")
    if coco_paths:
        print(f"   - Adnotacje COCO (DINO):   {coco_paths.get('train')}")
    print("=" * 75)
    print("\n>>> URUCHOMIENIE TRENINGU:")
    print(f"    YOLO: python src/train_evaluation/yolo_eval.py --epochs 60 --batch 16 --imgsz 640")
    print(f"    DINO: python src/train_evaluation/DINO_DEIM_eval.py --epochs 30 --batch-size 4\n")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Wagowanie danych i eksport do outputs/datasets/.")
    parser.add_argument("--concatenated-dir", type=Path, default=DEFAULT_CONCAT_DIR, help="Ścieżka do folderu data_concatenated")
    parser.add_argument("--good-weight", type=int, default=15, help="Waga dla zdjęć idealnych Felix_data (domyślnie: 15x)")
    parser.add_argument("--other-weight", type=int, default=1, help="Waga dla pozostałych zdjęć (domyślnie: 1x)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASETS_OUT, help="Katalog wyjściowy (domyślnie: outputs/datasets)")
    args = parser.parse_args()

    process_concatenated_dataset(
        concatenated_dir=args.concatenated_dir,
        good_weight=args.good_weight,
        other_weight=args.other_weight,
        output_datasets_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
