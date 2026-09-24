"""
File: src/models/model.py
Usage:
    from src.models.model import YOLOv8, DINOv3
Description:
    Main entry point for model architectures in the RagwortDetection project,
    exporting YOLOv8 and DINOv3 wrappers.
"""

from .yolov8 import YOLOv8

try:
    from .dinov3 import DINOv3
except ImportError:
    DINOv3 = None

__all__ = ["YOLOv8"]
if DINOv3 is not None:
    __all__.append("DINOv3")