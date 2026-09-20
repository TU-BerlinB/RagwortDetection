#!/usr/bin/env python3
"""Run one eval-only DEIMv2 + DINOv3 forward pass on a ragwort image.

Run from ``RagwortDetection`` after installing DEIMv2 requirements and placing
the official checkpoint at ``DEIMv2/ckpts/vitt_distill.pt``:
    python3 src/scripts/test_deimv2_dinov3.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEIMV2_ROOT = WORKSPACE_ROOT / "DEIMv2"
DEFAULT_CONFIG = DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_dinov3_ragwort.yml"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "outputs" / "combined_dataset_coco" / "val" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, help="Image to test; defaults to the first COCO val image.")
    return parser.parse_args()


def print_shapes(name: str, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        print(f"{name}: {tuple(value.shape)}")
    elif isinstance(value, (list, tuple)):
        print(f"{name}: {[tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__ for item in value]}")
    elif isinstance(value, dict):
        print(f"{name}: {{" + ", ".join(f"{key}={tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__}" for key, item in value.items()) + "}")
    else:
        print(f"{name}: {type(value).__name__}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not DEIMV2_ROOT.is_dir():
        raise FileNotFoundError(f"DEIMv2 repository not found: {DEIMV2_ROOT}")

    # Adapter checkpoint paths are relative to DEIMv2's working directory.
    os.chdir(DEIMV2_ROOT)
    sys.path.insert(0, str(DEIMV2_ROOT))
    import torch
    from PIL import Image
    from torchvision import transforms
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    weights_path = Path(cfg.yaml_cfg["DINOv3STAs"]["weights_path"])
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Required pretrained DINOv3/ViT checkpoint is missing: {weights_path.resolve()}\n"
            "Download the official checkpoint and place it at this path; refusing to test a randomly initialized backbone."
        )
    image_path = args.image.resolve() if args.image else next(iter(sorted(DEFAULT_IMAGE_DIR.glob("*"))), None)
    if image_path is None or not image_path.is_file():
        raise FileNotFoundError("No image found. Run yolo_to_coco.py first or pass --image PATH.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    original_size = torch.tensor([[image.height, image.width]], device=device)
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    inputs = preprocess(image).unsqueeze(0).to(device)
    model = cfg.model.to(device).eval()
    print(f"Device: {device}; image: {image_path}")
    with torch.no_grad():
        features = model.backbone(inputs)
        encoded = model.encoder(features)
        decoded = model.decoder(encoded)
        predictions = cfg.postprocessor(decoded, original_size)
    print_shapes("input", inputs)
    print_shapes("DINOv3STAs feature maps", features)
    print_shapes("HybridEncoder output", encoded)
    print_shapes("DEIMTransformer output", decoded)
    print_shapes("final predictions[0]", predictions[0])
    print("Forward pass successful")


if __name__ == "__main__":
    main()
