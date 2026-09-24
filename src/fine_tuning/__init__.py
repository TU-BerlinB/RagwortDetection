"""
Fine-tuning module for RagwortDetection.
Provides:
  - Stages 1 & 2: Two-stage fine-tuning (real data & weighted real data).
  - Stages 3, 4 & 5: Leaves & structure fine-tuning (1024 res, augmentations, no-yellow).
"""

from .fine_tuning import (
    FineTuningPipeline,
    prepare_fine_tuning_datasets,
    run_fine_tuning,
)
from .fine_tuning_leaves import (
    LeavesFineTuningPipeline,
    prepare_no_yellow_dataset,
    remove_yellow_from_image,
    run_leaves_fine_tuning,
)

__all__ = [
    "FineTuningPipeline",
    "prepare_fine_tuning_datasets",
    "run_fine_tuning",
    "LeavesFineTuningPipeline",
    "prepare_no_yellow_dataset",
    "remove_yellow_from_image",
    "run_leaves_fine_tuning",
]
