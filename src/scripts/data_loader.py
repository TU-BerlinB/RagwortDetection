"""
File: src/scripts/data_loader.py
Usage:
    from src.scripts.data_loader import get_dataloader, RagwortDataset
    loader = get_dataloader(data_dir="data/combined_dataset", split="train", batch_size=16)
    # or run self-test:
    python src/scripts/data_loader.py
Description:
    PyTorch Dataset and DataLoader factory for ragwort detection datasets in YOLO format,
    supporting bounding box coordinate transforms (xyxy pixels or normalized yolo format)
    and custom batch collation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "combined_dataset"


class RagwortDataset(Dataset):
    """
    PyTorch Dataset for ragwort detection datasets in YOLO format.

    Returns:
        image: torch.Tensor of shape [3, H, W]
        target: dict containing:
            - 'boxes': FloatTensor [N, 4] with bounding boxes
            - 'labels': LongTensor [N] with class IDs
            - 'image_id': int
            - 'filename': str (e.g. 'ragwort_001.jpg')
            - 'orig_size': torch.Tensor [2] (original height, width)
            - 'has_ragwort': bool (True if image contains >= 1 bounding box)
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
            data_dir: Dataset root directory (e.g. data/combined_dataset).
            split: Dataset partition: 'train', 'val', or 'test'.
            img_size: Target image dimensions (width, height), e.g. (640, 640).
            box_format: Bounding box coordinate format:
                        - 'xyxy': pixel coordinates [xmin, ymin, xmax, ymax]
                        - 'yolo': normalized coordinates [xc, yc, width, height] in [0, 1]
            transform: Optional torchvision transform pipeline.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.box_format = box_format.lower()

        if self.box_format not in ("xyxy", "yolo"):
            raise ValueError(f"Unknown box_format: {box_format}. Choose 'xyxy' or 'yolo'.")

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

        self.image_paths: List[Path] = []
        if self.images_dir.exists():
            self.image_paths = sorted(
                [p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            )

        self.class_names = self._load_class_names()

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
        """Parse YOLO label text file."""
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
                    xmin = (xc - bw / 2.0) * target_w
                    ymin = (yc - bh / 2.0) * target_h
                    xmax = (xc + bw / 2.0) * target_w
                    ymax = (yc + bh / 2.0) * target_h
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    boxes.append([xc, yc, bw, bh])

                labels.append(cls_id)

        return boxes, labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        img_path = self.image_paths[index]

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            orig_w, orig_h = image.size

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = self._parse_labels(label_path, orig_w, orig_h)

        image_tensor = self.transform(image)

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
    Collate function for object detection batches.
    Stacks images into [Batch, 3, H, W] and keeps variable-length targets in a tuple.
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
    Factory function creating a configured PyTorch DataLoader.

    Args:
        data_dir: Dataset root directory.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size.
        shuffle: Whether to shuffle data (defaults to True for train, False for val/test).
        num_workers: Number of DataLoader worker processes.
        img_size: Target image dimensions.
        box_format: 'xyxy' or 'yolo'.
        transform: Optional custom torchvision transforms.
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
    print("--- Testing data_loader.py ---")

    if not DEFAULT_DATA_DIR.exists() or not any(DEFAULT_DATA_DIR.iterdir()):
        print(f"[INFO] Dataset not found in: {DEFAULT_DATA_DIR}")
        print("[INFO] Creating temporary synthetic mini-dataset to verify DataLoader...")

        test_dir = REPO_ROOT / "data" / "_test_dataset"
        images_dir = test_dir / "train" / "images"
        labels_dir = test_dir / "train" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            dummy_img = Image.new("RGB", (640, 480), color=(30 * i, 100 + 20 * i, 50))
            dummy_img.save(images_dir / f"test_img_{i}.jpg")

            with open(labels_dir / f"test_img_{i}.txt", "w", encoding="utf-8") as f:
                f.write(f"0 {0.2 + 0.1 * i:.2f} {0.3 + 0.1 * i:.2f} 0.20 0.25\n")
                if i > 0:
                    f.write(f"0 0.70 0.60 0.15 0.20\n")

        loader = get_dataloader(data_dir=test_dir, split="train", batch_size=2, shuffle=False)
        print(f"[SUCCESS] DataLoader created. Number of samples: {len(loader.dataset)}")

        for batch_idx, (images, targets) in enumerate(loader):
            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"Images tensor shape [B, C, H, W]: {images.shape}")
            print(f"Number of targets in batch: {len(targets)}")
            for t_idx, target in enumerate(targets):
                print(f"  Image: {target['filename']}, boxes shape: {target['boxes'].shape}, has_ragwort: {target['has_ragwort']}")

        shutil.rmtree(test_dir, ignore_errors=True)
        print("\n[SUCCESS] data_loader.py self-test passed!")
    else:
        loader = get_dataloader(data_dir=DEFAULT_DATA_DIR, split="train", batch_size=4, num_workers=0)
        print(f"[INFO] Dataset found: {len(loader.dataset)} images in train split.")
        images, targets = next(iter(loader))
        print(f"Batch images shape: {images.shape}")
        print("First sample bounding boxes:", targets[0]["boxes"])