"""
yolov8.py

Klasa wrapper dla modelu YOLOv8 w projekcie detekcji starca (RagwortDetection).
Umożliwia:
  - Inicjalizację modelu z wagami bazowymi (np. yolov8n.pt) lub własnym checkpointem (.pt)
  - Trening modelu na zbiorze w formacie YOLO (np. data/combined_dataset/data.yaml)
  - Predykcję na pojedynczych obrazach, batchach i katalogach
  - Generowanie predykcji kompatybilnych z formatem ewaluacji w data_eval.py
  - Walidację i wyliczanie metryk (mAP50, mAP50-95, precision, recall)
  - Eksport do formatu ONNX / TorchScript
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Ścieżka bazowa projektu
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "combined_dataset" / "data.yaml"


class YOLOv8:
    """
    Wrapper dla modelu detekcji YOLOv8.

    Przykłady użycia:
        >>> from src.models.yolov8 import YOLOv8
        >>> model = YOLOv8("yolov8n.pt")
        >>> results = model.predict("sciezka/do/obrazu.jpg")
        >>> # Trening na połączonym zbiorze:
        >>> # model.train(data="data/combined_dataset/data.yaml", epochs=30)
    """

    def __init__(
        self,
        model_weight: Union[str, Path] = "yolov8n.pt",
        device: Optional[str] = None,
        task: str = "detect",
    ):
        """
        Args:
            model_weight: Nazwa bazowego modelu (np. 'yolov8n.pt', 'yolov8s.pt')
                          lub ścieżka do wytrenowanego pliku wag .pt.
            device: 'cuda', 'cpu' lub None (automatyczne wykrycie).
            task: Zadanie modelu ('detect', 'segment', 'classify'). Domyślnie 'detect'.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_weight = str(model_weight)
        self.task = task

        # Inicjalizacja modelu Ultralytics
        self.model = YOLO(self.model_weight, task=self.task)

        # Przeniesienie na odpowiednie urządzenie
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        self.best_weight_path: Optional[Path] = None

    def train(
        self,
        data: Union[str, Path] = DEFAULT_DATA_YAML,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        lr0: float = 0.005,
        workers: int = 2,
        project: Optional[str] = None,
        name: str = "ragwort_yolov8",
        exist_ok: bool = True,
        save: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Trenuje model YOLOv8 na podanym zbiorze danych.

        Args:
            data: Ścieżka do pliku data.yaml (np. data/combined_dataset/data.yaml).
            epochs: Liczba epok treningu.
            imgsz: Rozdzielczość obrazu wejściowego (domyślnie 640).
            batch: Rozmiar batcha.
            lr0: Początkowy współczynnik uczenia.
            project: Katalog nadrzędny wyników treningu (domyślnie runs).
            name: Nazwa folderu eksperymentu.
            exist_ok: Nadpisywanie istniejącego folderu eksperymentu.
            save: Zapisywanie wag checkpointów.
            verbose: Wypisywanie szczegółów treningu.
            **kwargs: Dodatkowe parametry przekazywane do model.train().

        Returns:
            Obiekt wyników treningu Ultralytics (results).
        """
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = (REPO_ROOT / data_path).resolve()

        if not data_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku konfiguracji zbioru: {data_path}.\n"
                "Uruchom najpierw: python src/scripts/data_download.py"
            )

        train_project = project or str(REPO_ROOT / "runs")

        print(f"[YOLOv8] Rozpoczynam trening na urządzeniu: {self.device}")
        print(f"[YOLOv8] Zbiór: {data_path} | Epoki: {epochs} | Batch: {batch} | Imgsz: {imgsz}")

        results = self.model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            workers=workers,
            device=self.device,
            project=train_project,
            name=name,
            exist_ok=exist_ok,
            save=save,
            verbose=verbose,
            **kwargs,
        )

        # Zapis ścieżki do najlepszych wag
        task_subfolder = "segment" if ("seg" in str(self.model_weight) or self.task == "segment") else "detect"
        trainer_dir = getattr(self.model.trainer, "save_dir", None)
        save_dir = Path(trainer_dir) if trainer_dir else Path(train_project) / task_subfolder / name
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            self.best_weight_path = best_pt
            print(f"[YOLOv8] Najlepsze wagi zapisano w: {self.best_weight_path}")

            # Kopiujemy wagi również do models/ dla łatwego dostępu
            models_dir = REPO_ROOT / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            saved_filename = "ragwort_yolov8_seg_best.pt" if task_subfolder == "segment" else "ragwort_yolov8_best.pt"
            saved_copy = models_dir / saved_filename
            import shutil
            shutil.copy2(best_pt, saved_copy)
            print(f"[YOLOv8] Kopia wag zapisana w: {saved_copy}")

        return results

    def predict(
        self,
        source: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, List[Any]],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Wykonuje predykcję detekcji lub segmentacji na obrazie lub liście obrazów.

        Args:
            source: Obraz (PIL, numpy, ścieżka do pliku, folder, tensor).
            conf: Próg ufności (confidence threshold, 0.0 - 1.0).
            iou: Próg IoU dla NMS (Non-Maximum Suppression).
            imgsz: Rozmiar wejściowy obrazu.
            verbose: Czy wypisywać logi inferencji.
            **kwargs: Dodatkowe argumenty dla model.predict().

        Returns:
            Lista słowników dla każdego przetworzonego obrazu:
            [
                {
                    "boxes": [[xmin, ymin, xmax, ymax], ...],  # w pikselach
                    "scores": [0.94, ...],                     # prawdopodobieństwa
                    "labels": [0, ...],                        # indeksy klas
                    "class_names": ["ragwort", ...],           # nazwy klas
                    "masks": [np.ndarray, ...],                # poligony masek (piksele)
                    "masks_normalized": [np.ndarray, ...],     # poligony znormalizowane (0..1)
                    "orig_shape": (wysokość, szerokość),
                    "raw": Obiekt Results z ultralytics
                },
                ...
            ]
        """
        raw_results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=verbose,
            **kwargs,
        )

        parsed_output: List[Dict[str, Any]] = []

        for res in raw_results:
            boxes_xyxy: List[List[float]] = []
            scores: List[float] = []
            labels: List[int] = []
            class_names: List[str] = []
            masks_xy: List[np.ndarray] = []
            masks_xyn: List[np.ndarray] = []

            if res.boxes is not None and len(res.boxes) > 0:
                # boxes w formacie [xmin, ymin, xmax, ymax]
                boxes_xyxy = res.boxes.xyxy.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()

                # Mapowanie ID klasy na nazwę
                names_dict = res.names or {}
                class_names = [names_dict.get(cls_id, str(cls_id)) for cls_id in labels]

            if res.masks is not None:
                masks_xy = [p for p in res.masks.xy]
                masks_xyn = [p for p in res.masks.xyn]

            parsed_output.append(
                {
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": labels,
                    "class_names": class_names,
                    "masks": masks_xy,
                    "masks_normalized": masks_xyn,
                    "orig_shape": res.orig_shape,
                    "raw": res,
                }
            )

        return parsed_output

    def predict_for_eval(
        self,
        images_dir: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, List[List[float]]]:
        """
        Generuje słownik predykcji bezpośrednio kompatybilny z modułem data_eval.py:
            { "nazwa_zdjecia.jpg": [[xmin, ymin, xmax, ymax], ...] }

        Args:
            images_dir: Ścieżka do katalogu ze zdjęciami testowymi.
            conf: Próg ufności.
            iou: Próg NMS.

        Returns:
            Słownik: nazwa_pliku -> lista ramek [xmin, ymin, xmax, ymax].
        """
        img_dir = Path(images_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"Katalog zdjęć nie istnieje: {img_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in sorted(img_dir.iterdir())
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        predictions_for_eval: Dict[str, List[List[float]]] = {}

        for img_file in image_files:
            preds = self.predict(img_file, conf=conf, iou=iou, verbose=False)
            if preds and len(preds) > 0:
                predictions_for_eval[img_file.name] = preds[0]["boxes"]
            else:
                predictions_for_eval[img_file.name] = []

        return predictions_for_eval

    def val(
        self,
        data: Union[str, Path] = DEFAULT_DATA_YAML,
        imgsz: int = 640,
        batch: int = 16,
        split: str = "val",
        verbose: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Wykonuje walidację modelu i zwraca metryki (mAP50, mAP50-95 itp.).
        """
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = (REPO_ROOT / data_path).resolve()

        return self.model.val(
            data=str(data_path),
            imgsz=imgsz,
            batch=batch,
            split=split,
            device=self.device,
            verbose=verbose,
            **kwargs,
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Eksportuje model do formatu np. 'onnx', 'torchscript'."""
        return self.model.export(format=format, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Pozwala wywołać instancję jak funkcję: model(obraz)."""
        return self.predict(*args, **kwargs)


if __name__ == "__main__":
    print("=== Testowanie klasy YOLOv8 ===")
    yolo = YOLOv8(model_weight="yolov8n.pt")
    print(f"Urządzenie modelu: {yolo.device}")

    # Test predykcji na syntetycznym obrazie
    dummy_image = Image.new("RGB", (640, 640), color=(50, 120, 50))
    res = yolo.predict(dummy_image)

    print("Inferencja zakończona sukcesem!")
    print("Liczba wykrytych ramek:", len(res[0]["boxes"]))
    print("Kształt wejściowy:", res[0]["orig_shape"])
