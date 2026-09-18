"""
model.py

Główny punkt dostępowy do modeli w projekcie RagwortDetection.
Eksportuje modele YOLOv8 oraz DINOv3.
"""

from .yolov8 import YOLOv8
from .dinov3 import DINOv3

__all__ = ["YOLOv8", "DINOv3"]