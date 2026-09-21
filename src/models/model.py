"""
model.py

Główny punkt dostępowy do modeli w projekcie RagwortDetection.
Eksportuje modele YOLOv8 oraz DINOv3.
"""

from .yolov8 import YOLOv8

try:
    from .dinov3 import DINOv3
except ImportError:
    DINOv3 = None

__all__ = ["YOLOv8"]
if DINOv3 is not None:
    __all__.append("DINOv3")