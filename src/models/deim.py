"""
deim.py

Wrapper dla modelu detekcji DEIMv2 (DINOv3 + DEIM Transformer).
Zapewnia interfejs kompatybilny z pipeline'em YOLOv8 i aplikacją webową.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DEIMV2_ROOT = REPO_ROOT.parent / "DEIMv2"
if str(DEIMV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DEIMV2_ROOT))

from engine.core.workspace import create
from engine.core.yaml_utils import merge_config


class DEIMBoxes:
    """Wrapper dla ramek detekcji emulujący strukturę ultralytics Boxes."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def cpu(self) -> DEIMBoxes:
        return DEIMBoxes(self.xyxy.cpu(), self.conf.cpu(), self.cls.cpu())

    def numpy(self) -> DEIMBoxes:
        return self


class DEIMResult:
    """Wynik pojedynczej predykcji emulujący interfejs Results z ultralytics."""

    def __init__(
        self,
        boxes: DEIMBoxes,
        orig_shape: tuple[int, int],
        names: dict[int, str],
    ):
        self.boxes = boxes
        self.masks = None
        self.orig_shape = orig_shape
        self.names = names


class DEIMModel:
    """
    Wrapper dla modelu DEIMv2 + DINOv3.
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = "outputs/models/model.pt",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wp = Path(weights_path)
        if not wp.is_absolute():
            candidates = [
                REPO_ROOT / "outputs" / "models" / wp.name,
                REPO_ROOT / "outputs" / "weights" / wp.name,
                REPO_ROOT / "outputs" / wp,
                REPO_ROOT / wp,
            ]
            for c in candidates:
                if c.exists():
                    self.weights_path = c.resolve()
                    break
            else:
                self.weights_path = (REPO_ROOT / wp).resolve()
        else:
            self.weights_path = wp

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Nie znaleziono wag DEIM: {self.weights_path}")

        print(f"[DEIM] Ładowanie wag DEIMv2 + DINOv3 z {self.weights_path} na {self.device}...")

        # Załadowanie checkpointu
        data = torch.load(self.weights_path, map_location="cpu")
        cfg = data["config"]
        global_cfg = merge_config(cfg, inplace=False, overwrite=False)

        # Inicjalizacja modelu i postprocessora
        self.model = create(cfg["model"], global_cfg)
        self.model.load_state_dict(data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.postprocessor = create(cfg["postprocessor"], global_cfg)
        self.categories = data.get("categories", [{"id": 1, "name": "ragwort"}])
        self.names = {c.get("id", 1): c.get("name", "ragwort") for c in self.categories}
        self.names[0] = "ragwort"

        self.task = "detect"
        self.model_weight = str(self.weights_path)

        # Transformacje wejściowe standardowe dla ViT / DEIM
        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        print(f"[DEIM] Model DEIMv2 gotowy do pracy! Klasy: {self.names}")

    def predict(
        self,
        source: Union[str, Path, Image.Image, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[DEIMResult]:
        """
        Wykonuje predykcję na obrazie i zwraca listę DEIMResult.
        """
        # Przygotowanie obrazu PIL i odczyt wymiarów
        if isinstance(source, (str, Path)):
            img_path = Path(source)
            if not img_path.exists():
                raise FileNotFoundError(f"Nie znaleziono pliku: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(source, Image.Image):
            pil_img = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            # OpenCV BGR -> RGB
            if len(source.shape) == 3 and source.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(source)
        else:
            raise TypeError(f"Nieobsługiwany typ źródła obrazu: {type(source)}")

        w, h = pil_img.size
        orig_size = torch.tensor([[h, w]], device=self.device)

        inputs = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            preds = self.postprocessor(outputs, orig_size)[0]

        scores = preds["scores"]
        keep = scores >= conf
        filtered_boxes = preds["boxes"][keep]
        filtered_scores = scores[keep]
        filtered_labels = preds["labels"][keep]

        if len(filtered_boxes) > 0 and iou > 0:
            nms_idx = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=iou)
            final_boxes = filtered_boxes[nms_idx]
            final_scores = filtered_scores[nms_idx]
            final_labels = filtered_labels[nms_idx]
        else:
            final_boxes = filtered_boxes
            final_scores = filtered_scores
            final_labels = filtered_labels

        boxes_wrapper = DEIMBoxes(
            xyxy=final_boxes,
            conf=final_scores,
            cls=final_labels,
        )

        result = DEIMResult(
            boxes=boxes_wrapper,
            orig_shape=(h, w),
            names=self.names,
        )

        if verbose:
            print(f"[DEIM] Wykryto {len(final_boxes)} obiektów (conf >= {conf})")

        return [result]

    def __call__(self, *args: Any, **kwargs: Any) -> List[DEIMResult]:
        return self.predict(*args, **kwargs)
