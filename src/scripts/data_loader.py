"""
data_loader.py

Moduł ładowania danych dla projektu detekcji starca (RagwortDetection).
Obsługuje format YOLO (przygotowany przez data_download.py) zawierający:
  - katalogi 'images/' ze zdjęciami (.jpg, .png, etc.)
  - katalogi 'labels/' z plikami tekstowymi (.txt) z ramkami (bounding boxes)

Zawiera:
  - RagwortDataset: klasa PyTorch Dataset obsługująca obrazy i ramki YOLO.
  - collate_fn: funkcja grupująca próbki dla detekcji obiektów (zmienna liczba ramek).
  - get_dataloader: fabryka do szybkiego tworzenia DataLoaderów (train / val / test).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

# Rozszerzenia plików graficznych
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Ścieżka do katalogu głównego projektu
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "combined_dataset"


class RagwortDataset(Dataset):
    """
    PyTorch Dataset dla zbioru detekcji starca w formacie YOLO.

    Zwraca:
        image: torch.Tensor o kształcie [3, H, W]
        target: dict z polami:
            - 'boxes': FloatTensor [N, 4] z ramkami ograniczającymi
            - 'labels': LongTensor [N] z indeksami klas
            - 'image_id': int
            - 'filename': str (np. 'ragwort_001.jpg')
            - 'orig_size': torch.Tensor [2] (wysokość, szerokość oryginału)
            - 'has_ragwort': bool (True jeśli zdjęcie zawiera co najmniej jedną ramkę)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        split: str = "train",
        img_size: Optional[Tuple[int, int]] = (640, 640),
        box_format: str = "xyxy",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_dir: Ścieżka do katalogu ze zbiorem (np. data/combined_dataset).
            split: Podzbiór: 'train', 'val' lub 'test'.
            img_size: Docelowy rozmiar obrazu (szerokość, wysokość), np. (640, 640) lub (224, 224).
            box_format: Format zwracanych ramek:
                        - 'xyxy': piksele docelowe [xmin, ymin, xmax, ymax] (kompatybilne z data_eval.py)
                        - 'yolo': znormalizowane [xc, yc, width, height] w przedziale [0, 1]
            transform: Opcjonalne transformacje torchvision (jeśli None, używany jest domyślny resize + to_tensor).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.box_format = box_format.lower()

        if self.box_format not in ("xyxy", "yolo"):
            raise ValueError(f"Nieznany box_format: {box_format}. Wybierz 'xyxy' lub 'yolo'.")

        # Elastyczne wykrywanie ścieżek:
        # 1. data_dir / split / images (standard data_download.py)
        # 2. data_dir / images (gdy wskazano bezpośrednio folder splitu)
        if (self.data_dir / self.split / "images").exists():
            self.images_dir = self.data_dir / self.split / "images"
            self.labels_dir = self.data_dir / self.split / "labels"
        elif (self.data_dir / "images").exists():
            self.images_dir = self.data_dir / "images"
            self.labels_dir = self.data_dir / "labels"
        elif self.data_dir.exists() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in self.data_dir.iterdir() if p.is_file()):
            self.images_dir = self.data_dir
            self.labels_dir = self.data_dir.parent / "labels"
        else:
            self.images_dir = self.data_dir / self.split / "images"
            self.labels_dir = self.data_dir / self.split / "labels"

        # Znajdź wszystkie pliki graficzne
        self.image_paths: List[Path] = []
        if self.images_dir.exists():
            self.image_paths = sorted(
                [p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            )

        # Wczytanie nazw klas z data.yaml (jeśli plik istnieje)
        self.class_names = self._load_class_names()

        # Domyślne transformacje
        if transform is not None:
            self.transform = transform
        else:
            t_list = []
            if self.img_size is not None:
                t_list.append(transforms.Resize(self.img_size, antialias=True))
            t_list.append(transforms.ToTensor())
            self.transform = transforms.Compose(t_list)

    def _load_class_names(self) -> List[str]:
        yaml_path = self.data_dir / "data.yaml"
        if not yaml_path.exists():
            yaml_path = self.data_dir.parent / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                names = data.get("names")
                if isinstance(names, dict):
                    return [names[i] for i in sorted(names, key=int)]
                if isinstance(names, list):
                    return [str(n) for n in names]
            except Exception:
                pass
        return ["ragwort"]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_labels(self, label_path: Path, orig_w: int, orig_h: int) -> Tuple[List[List[float]], List[int]]:
        """Parsuje plik etykiet YOLO."""
        boxes: List[List[float]] = []
        labels: List[int] = []

        if not label_path.exists():
            return boxes, labels

        target_w = self.img_size[0] if self.img_size else orig_w
        target_h = self.img_size[1] if self.img_size else orig_h

        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                if self.box_format == "xyxy":
                    # Konwersja na piksele w przestrzeni docelowego obrazu [xmin, ymin, xmax, ymax]
                    xmin = (xc - bw / 2.0) * target_w
                    ymin = (yc - bh / 2.0) * target_h
                    xmax = (xc + bw / 2.0) * target_w
                    ymax = (yc + bh / 2.0) * target_h
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    # Format YOLO: [xc, yc, bw, bh] znormalizowany (0-1)
                    boxes.append([xc, yc, bw, bh])

                labels.append(cls_id)

        return boxes, labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        img_path = self.image_paths[index]

        # 1. Bezpieczne wczytanie obrazu i konwersja do 3-kanałowego RGB
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            orig_w, orig_h = image.size

        # 2. Wczytanie i przeliczenie ramki z pliku labels/.txt
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = self._parse_labels(label_path, orig_w, orig_h)

        # 3. Zastosowanie transformacji obrazu
        image_tensor = self.transform(image)

        # 4. Przygotowanie tensorów ramek i etykiet
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "filename": img_path.name,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "has_ragwort": len(boxes) > 0,
        }

        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Tuple[Dict, ...]]:
    """
    Dedykowany collate_fn dla detekcji obiektów.

    Standardowy DataLoader nie potrafi złączyć tensorów 'boxes' i 'labels' o różnej
    długości (różna liczba obiektów na zdjęciu).
    Ta funkcja:
      - łączy obrazy w jeden tensor [Batch_Size, 3, H, W]
      - zachowuje cele (targets) jako krotkę słowników o długości Batch_Size.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = tuple(item[1] for item in batch)
    return images, targets


def get_dataloader(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (640, 640),
    box_format: str = "xyxy",
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Fabryka ułatwiająca utworzenie skonfigurowanego DataLoadera.

    Args:
        data_dir: Katalog ze zbiorem danych.
        split: 'train', 'val' lub 'test'.
        batch_size: Rozmiar batcha.
        shuffle: Czy tasować dane (domyślnie: True dla train, False dla val/test).
        num_workers: Liczba procesów ładujących (na Windows bezpiecznie dać 0 lub 2 wewnątrz __main__).
        img_size: Rozmiar wejściowy obrazu.
        box_format: 'xyxy' lub 'yolo'.
        transform: Opcjonalne własne transformacje.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = RagwortDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        box_format=box_format,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    print("--- Testowanie modułu data_loader.py ---")

    # Sprawdzenie, czy zbiór istnieje na dysku
    if not DEFAULT_DATA_DIR.exists() or not any(DEFAULT_DATA_DIR.iterdir()):
        print(f"[INFO] Brak pobranego zbioru w: {DEFAULT_DATA_DIR}")
        print("[INFO] Tworzę tymczasowy mini-zbiór w data/_test_dataset do testu poprawnego działania...")

        test_dir = REPO_ROOT / "data" / "_test_dataset"
        images_dir = test_dir / "train" / "images"
        labels_dir = test_dir / "train" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Generujemy 3 syntetyczne zdjęcia testowe
        for i in range(3):
            dummy_img = Image.new("RGB", (640, 480), color=(30 * i, 100 + 20 * i, 50))
            dummy_img.save(images_dir / f"test_img_{i}.jpg")

            # Przykładowe etykiety YOLO (1 lub 2 ramki)
            with open(labels_dir / f"test_img_{i}.txt", "w", encoding="utf-8") as f:
                f.write(f"0 {0.2 + 0.1 * i:.2f} {0.3 + 0.1 * i:.2f} 0.20 0.25\n")
                if i > 0:
                    f.write(f"0 0.70 0.60 0.15 0.20\n")

        # Uruchomienie DataLoadera na wygenerowanym mini-zbiorze
        loader = get_dataloader(data_dir=test_dir, split="train", batch_size=2, shuffle=False)
        print(f"[SUKCES] Utworzono DataLoader. Liczba próbek w zbiorze: {len(loader.dataset)}")

        for batch_idx, (images, targets) in enumerate(loader):
            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"Kształt tensora obrazów [B, C, H, W]: {images.shape}")
            print(f"Liczba celów (targets) w batchu: {len(targets)}")
            for t_idx, target in enumerate(targets):
                print(f"  Zdjęcie: {target['filename']}, ramki (boxes shape): {target['boxes'].shape}, ma starca: {target['has_ragwort']}")
                if len(target["boxes"]) > 0:
                    print(f"    Pierwsza ramka (xyxy piksele): {target['boxes'][0].tolist()}")

        # Sprzątanie folderu testowego
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        print("\n[SUKCES] data_loader.py działa bezbłędnie!")
    else:
        # Jeśli zbiór istnieje, testujemy na prawdziwych danych
        loader = get_dataloader(data_dir=DEFAULT_DATA_DIR, split="train", batch_size=4, num_workers=0)
        print(f"[INFO] Znaleziono zbiór danych: {len(loader.dataset)} zdjęć w podzbiorze train.")
        images, targets = next(iter(loader))
        print(f"Kształt batcha obrazów: {images.shape}")
        print("Przykładowe ramki pierwszego zdjęcia:", targets[0]["boxes"])