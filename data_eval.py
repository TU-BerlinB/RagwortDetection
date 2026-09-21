"""
data_eval.py - Glowny punkt startowy treningu i ewaluacji modelu YOLOv8.

Uzycie:
  # Uruchomienie na GPU (CUDA) z domyslnym autowykryciem zwagowanego data_weighted.yaml:
  python data_eval.py --epochs 50 --batch 16

  # Wskazanie wlasnego pliku danych:
  python data_eval.py --data data/data_concatenated/data_weighted.yaml --epochs 50 --batch 16 --device 0

  # Tylko ewaluacja istniejacych wag (bez treningu):
  python data_eval.py --skip-train --weights models/ragwort_yolov8_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_evaluation.yolo_eval import main

if __name__ == "__main__":
    main()
