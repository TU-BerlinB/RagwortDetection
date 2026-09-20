"""
train_segmentation.py
Skrot do uruchamiania treningu segmentacji organow starca jakubka.
Uzycie:
  python scripts/train_segmentation.py --help
  python scripts/train_segmentation.py --model yolov8n-seg.pt --epochs 50
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_evaluation.train_ragwort_segmentation import main

if __name__ == "__main__":
    main()
