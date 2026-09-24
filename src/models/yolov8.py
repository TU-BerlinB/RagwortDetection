"""
File: src/models/yolov8.py
Usage:
    from src.models.yolov8 import YOLOv8
    model = YOLOv8("yolov8s.pt")
    results = model.predict("path/to/image.jpg")
    # or run self-test:
    python src/models/yolov8.py
Description:
    Wrapper for Ultralytics YOLOv8 object detection and segmentation,
    supporting model initialization, training on YOLO format datasets,
    batch prediction, metric validation, and model export.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "combined_dataset" / "data.yaml"

# Configure Ultralytics settings so runs and weights are stored in outputs/
try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str((REPO_ROOT / "outputs" / "weights").resolve()),
        "runs_dir": str((REPO_ROOT / "outputs" / "runs").resolve()),
        "datasets_dir": str((REPO_ROOT / "outputs" / "datasets").resolve()),
    })
except Exception:
    pass


class YOLOv8:
    """
    Wrapper for YOLOv8 object detection and segmentation model.

    Examples:
        >>> from src.models.yolov8 import YOLOv8
        >>> model = YOLOv8("yolov8s.pt")
        >>> results = model.predict("path/to/image.jpg")
    """

    def __init__(
        self,
        model_weight: Union[str, Path] = "yolov8s.pt",
        device: Optional[str] = None,
        task: str = "detect",
    ):
        """
        Args:
            model_weight: Name of the base model (e.g. 'yolov8s.pt') or path to a trained checkpoint.
            device: 'cuda', 'cuda:0', 'cpu', or None for automatic detection.
            task: Task type ('detect', 'segment', 'classify'). Defaults to 'detect'.
        """
        # Normalize device string for PyTorch
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif str(device).isdigit():
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        elif str(device).lower() in ("gpu", "cuda"):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Resolve model weights location
        mw = Path(model_weight)
        weights_dir = REPO_ROOT / "outputs" / "weights"
        models_dir = REPO_ROOT / "outputs" / "models"
        outputs_dir = REPO_ROOT / "outputs"
        weights_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        if mw.is_file() and mw.parent.resolve() != (REPO_ROOT / "weights").resolve():
            self.model_weight = str(mw.resolve())
        elif (weights_dir / mw.name).is_file():
            self.model_weight = str((weights_dir / mw.name).resolve())
        elif (models_dir / mw.name).is_file():
            self.model_weight = str((models_dir / mw.name).resolve())
        elif (outputs_dir / mw.name).is_file():
            self.model_weight = str((outputs_dir / mw.name).resolve())
        elif (REPO_ROOT / "weights" / mw.name).is_file():
            target_pt = weights_dir / mw.name
            shutil.copy2(REPO_ROOT / "weights" / mw.name, target_pt)
            self.model_weight = str(target_pt.resolve())
        elif mw.suffix in (".pt", ".pth") or not mw.parent.name:
            self.model_weight = str((weights_dir / mw.name).resolve())
        else:
            self.model_weight = str(model_weight)
        self.task = task

        self.model = YOLO(self.model_weight, task=self.task)

        # Ensure base weight file is accessible in outputs/
        try:
            resolved_weight = Path(self.model_weight)
            if resolved_weight.is_file() and resolved_weight.parent.resolve() == weights_dir.resolve():
                outputs_link = outputs_dir / resolved_weight.name
                if not outputs_link.exists():
                    try:
                        outputs_link.symlink_to(Path("weights") / resolved_weight.name)
                    except Exception:
                        shutil.copy2(resolved_weight, outputs_link)
        except Exception:
            pass

        try:
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception:
            pass

        self.best_weight_path: Optional[Path] = None

    def train(
        self,
        data: Union[str, Path] = DEFAULT_DATA_YAML,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        lr0: float = 0.005,
        workers: int = 2,
        device: Optional[str] = None,
        project: Optional[str] = None,
        name: str = "ragwort_yolov8",
        exist_ok: bool = True,
        save: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Train the YOLOv8 model on a specified dataset.

        Args:
            data: Path to data.yaml dataset definition.
            epochs: Total number of training epochs.
            imgsz: Target image input size (default 640).
            batch: Training batch size.
            lr0: Initial learning rate.
            workers: DataLoader worker threads.
            device: Computation device ('cuda:0', 'cpu', etc.).
            project: Directory to save training runs.
            name: Experiment subfolder name.
            exist_ok: Overwrite existing experiment folder.
            save: Save checkpoints during training.
            verbose: Verbose training output.
            **kwargs: Extra arguments forwarded to model.train().

        Returns:
            Ultralytics training results object.
        """
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = (REPO_ROOT / data_path).resolve()

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found: {data_path}.\n"
                "Run data preparation first, e.g.: python src/scripts/weight_dataset.py"
            )

        train_project = project or str(REPO_ROOT / "outputs" / "runs" / "yolo")
        target_device = device if device is not None else self.device
        kwargs.pop("device", None)

        print(f"[YOLOv8] Starting training on device: {target_device}")
        print(f"[YOLOv8] Dataset: {data_path} | Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")

        results = self.model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            workers=workers,
            device=target_device,
            project=train_project,
            name=name,
            exist_ok=exist_ok,
            save=save,
            verbose=verbose,
            **kwargs,
        )

        task_subfolder = "segment" if ("seg" in str(self.model_weight) or self.task == "segment") else "detect"
        trainer_dir = getattr(self.model.trainer, "save_dir", None)
        save_dir = Path(trainer_dir) if trainer_dir else Path(train_project) / task_subfolder / name
        best_pt = save_dir / "weights" / "best.pt"

        if best_pt.exists():
            self.best_weight_path = best_pt
            print(f"[YOLOv8] Best weights saved to: {self.best_weight_path}")

            models_dir = REPO_ROOT / "outputs" / "models"
            weights_dir = REPO_ROOT / "outputs" / "weights"
            models_dir.mkdir(parents=True, exist_ok=True)
            weights_dir.mkdir(parents=True, exist_ok=True)
            saved_filename = "ragwort_yolov8_seg_best.pt" if task_subfolder == "segment" else "ragwort_yolov8_best.pt"
            saved_copy = models_dir / saved_filename

            shutil.copy2(best_pt, saved_copy)
            shutil.copy2(best_pt, weights_dir / saved_filename)
            shutil.copy2(best_pt, weights_dir / "best.pt")
            print(f"[YOLOv8] Copies saved to: {saved_copy} and {weights_dir}")

            root_w = REPO_ROOT / "weights"
            if root_w.is_dir() and root_w.resolve() != weights_dir.resolve():
                for f in root_w.iterdir():
                    if f.is_file():
                        shutil.copy2(f, weights_dir / f.name)
                try:
                    shutil.rmtree(root_w)
                except Exception:
                    pass

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
        Run object detection or segmentation inference on an image or batch of images.

        Args:
            source: Image input (PIL Image, numpy array, filepath, directory, tensor).
            conf: Confidence threshold (0.0 to 1.0).
            iou: IoU threshold for Non-Maximum Suppression (NMS).
            imgsz: Input image resolution.
            verbose: Enable verbose logging.
            **kwargs: Additional arguments for model.predict().

        Returns:
            List of parsed prediction dictionaries per image.
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
                boxes_xyxy = res.boxes.xyxy.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()

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
        Generate predictions dictionary mapping filename to bounding boxes:
            { "image_filename.jpg": [[xmin, ymin, xmax, ymax], ...] }

        Args:
            images_dir: Directory containing test images.
            conf: Confidence threshold.
            iou: NMS IoU threshold.

        Returns:
            Dictionary mapping image filename to list of bounding boxes.
        """
        img_dir = Path(images_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

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
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run validation on a dataset split and return validation metrics."""
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = (REPO_ROOT / data_path).resolve()

        target_device = device if device is not None else self.device
        kwargs.pop("device", None)

        return self.model.val(
            data=str(data_path),
            imgsz=imgsz,
            batch=batch,
            split=split,
            device=target_device,
            verbose=verbose,
            **kwargs,
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export model to target format (e.g. 'onnx', 'torchscript')."""
        return self.model.export(format=format, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.predict(*args, **kwargs)


if __name__ == "__main__":
    print("=== YOLOv8 Model Initialization Test ===")
    yolo = YOLOv8(model_weight="yolov8s.pt")
    print(f"Model device: {yolo.device}")

    dummy_image = Image.new("RGB", (640, 640), color=(50, 120, 50))
    res = yolo.predict(dummy_image)

    print("Inference completed successfully!")
    print("Detected boxes:", len(res[0]["boxes"]))
    print("Original input shape:", res[0]["orig_shape"])