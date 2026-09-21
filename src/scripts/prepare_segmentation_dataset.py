"""
prepare_segmentation_dataset.py

Ekstrahuje obrazy posiadające adnotacje poligonowe (segmentacyjne) z data/combined_dataset
i tworzy dedykowany, poprawny zbiór dla YOLOv8-seg w data/ragwort_segmentation.
"""

import glob
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "data" / "combined_dataset"
OUT_DIR = REPO_ROOT / "data" / "ragwort_segmentation"
YAML_PATH = REPO_ROOT / "data" / "ragwort_segmentation.yaml"


def is_pure_segmentation(label_file: Path) -> bool:
    """Sprawdza, czy wszystkie obiekty w pliku etykiety to poligony (więcej niż 4 współrzędne)."""
    with open(label_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]
    if not lines:
        return False
    return all(len(parts) > 5 for parts in lines)


def build_segmentation_dataset():
    print(f"[1/3] Przeszukiwanie {SRC_DIR} w poszukiwaniu adnotacji poligonowych...")

    for split in ["train", "test"]:
        target_split = "train" if split == "train" else "val"
        images_out = OUT_DIR / target_split / "images"
        labels_out = OUT_DIR / target_split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        labels_in = list((SRC_DIR / split / "labels").glob("*.txt"))
        seg_count = 0
        total_objects = 0

        for lbl_path in labels_in:
            if is_pure_segmentation(lbl_path):
                # Szukamy odpowiadającego obrazu
                img_name = lbl_path.stem
                found_img = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = SRC_DIR / split / "images" / f"{img_name}{ext}"
                    if candidate.exists():
                        found_img = candidate
                        break

                if found_img is not None:
                    # Kopiujemy parę obraz + etykieta
                    shutil.copy2(found_img, images_out / found_img.name)
                    shutil.copy2(lbl_path, labels_out / lbl_path.name)
                    seg_count += 1
                    with open(lbl_path) as fp:
                        total_objects += sum(1 for line in fp if line.strip())

        print(f" -> {target_split.upper()}: Skopiowano {seg_count} obrazów z {total_objects} obiektami poligonowymi.")

    print(f"[2/3] Tworzenie pliku konfiguracyjnego {YAML_PATH}...")
    yaml_content = (
        "# ragwort_segmentation.yaml\n"
        "# Zbiór danych do segmentacji instancji starca jakubka (Ragwort Instance Segmentation)\n"
        "path: data/ragwort_segmentation\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: val/images\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: ragwort\n"
        "  1: objects\n"
    )
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[3/3] Zbiór danych przygotowany pomyślnie w: {OUT_DIR}")


if __name__ == "__main__":
    build_segmentation_dataset()
