#!/usr/bin/env python3
"""Train or evaluate the Ragwort DEIMv2 + DINOv3 detector.

This is deliberately a thin project entry point: DEIMv2's own YAMLConfig,
DetSolver, criterion, COCO evaluator, and checkpoint implementation do the
actual work. In particular, losses are computed from raw decoder outputs,
never from postprocessed detections.

Examples (run from RagwortDetection):
  python src/scripts/train_deimv2_dinov3.py
  python src/scripts/train_deimv2_dinov3.py --epochs 10 --batch-size 4
  python src/scripts/train_deimv2_dinov3.py --resume outputs/checkpoints/deimv2_dinov3_ragwort/last.pth
  python src/scripts/train_deimv2_dinov3.py --evaluate --resume outputs/checkpoints/deimv2_dinov3_ragwort/best_stg1.pth
  python src/scripts/train_deimv2_dinov3.py --smoke-test --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEIMV2_ROOT = WORKSPACE_ROOT / "DEIMv2"
DEFAULT_CONFIG = DEIMV2_ROOT / "configs" / "deimv2" / "deimv2_dinov3_ragwort.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="DEIMv2 YAML configuration",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the YAML `epoches` value",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Total train/validation batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Detection-component learning rate",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="DataLoader worker count",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        help="Full DEIMv2 checkpoint to resume/evaluate",
    )

    parser.add_argument(
        "--device",
        help="Torch device, e.g. cuda, cuda:0, or cpu",
    )

    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Unfreeze the pretrained DINOv3 ViT",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run COCO validation only",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one raw-output forward/loss/backward/step and one-batch COCO evaluation",
    )

    return parser.parse_args()


def update_items(args: argparse.Namespace) -> list[str]:
    """Translate friendly arguments into DEIMv2 YAML overrides."""

    coco_root = PROJECT_ROOT / "outputs" / "combined_dataset_coco"

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "deimv2_dinov3_ragwort"
    )

    items: list[str] = [
        f"output_dir={output_dir}",
        f"train_dataloader.dataset.img_folder={coco_root / 'train' / 'images'}",
        f"train_dataloader.dataset.ann_file={coco_root / 'train' / 'annotations.json'}",
        f"val_dataloader.dataset.img_folder={coco_root / 'val' / 'images'}",
        f"val_dataloader.dataset.ann_file={coco_root / 'val' / 'annotations.json'}",
        f"test_dataloader.dataset.img_folder={coco_root / 'test' / 'images'}",
        f"test_dataloader.dataset.ann_file={coco_root / 'test' / 'annotations.json'}",
    ]

    if args.epochs is not None:
        items.append(f"epoches={args.epochs}")

    if args.batch_size is not None:
        items.extend(
            (
                f"train_dataloader.total_batch_size={args.batch_size}",
                f"val_dataloader.total_batch_size={args.batch_size}",
            )
        )

    if args.lr is not None:
        items.append(f"optimizer.lr={args.lr}")

    if args.workers is not None:
        items.extend(
            (
                f"train_dataloader.num_workers={args.workers}",
                f"val_dataloader.num_workers={args.workers}",
            )
        )

    if args.device is not None:
        items.append(f"device={args.device}")

    if args.resume is not None:
        items.append(f"resume={args.resume.resolve()}")

    if args.train_backbone:
        items.append("DINOv3STAs.finetune=True")

    return items


def require_paths(config_path: Path) -> None:
    if not DEIMV2_ROOT.is_dir():
        raise FileNotFoundError(
            f"DEIMv2 repository is required at {DEIMV2_ROOT}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )


def dataset_summary(
    config: dict[str, Any],
) -> list[tuple[str, int, int, list[dict[str, Any]]]]:

    summaries = []

    for split, loader_name in (
        ("train", "train_dataloader"),
        ("validation", "val_dataloader"),
    ):
        ann_path = Path(
            config[loader_name]["dataset"]["ann_file"]
        )

        if not ann_path.is_file():
            raise FileNotFoundError(
                f"{split} annotation file not found: {ann_path}"
            )

        data = json.loads(
            ann_path.read_text(encoding="utf-8")
        )

        summaries.append(
            (
                split,
                len(data.get("images", [])),
                len(data.get("annotations", [])),
                data.get("categories", []),
            )
        )

    return summaries


def startup_report(
    config_path: Path,
    overrides: list[str],
) -> None:

    sys.path.insert(0, str(DEIMV2_ROOT))

    from engine.core import YAMLConfig, yaml_utils

    config = YAMLConfig(
        str(config_path),
        **yaml_utils.parse_cli(overrides),
    )

    raw = config.yaml_cfg

    weights_path = Path(
        raw["DINOv3STAs"]["weights_path"]
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Required pretrained DINOv3 checkpoint is missing: "
            f"{weights_path.resolve()}\n"
            "Refusing to initialize the DINOv3 backbone randomly."
        )

    summaries = dataset_summary(raw)

    categories = summaries[0][3]

    if not categories:
        raise ValueError(
            "The COCO training annotation file has no categories."
        )

    category_ids = [
        category["id"]
        for category in categories
    ]

    if raw["num_classes"] != len(categories):
        raise ValueError(
            f"num_classes={raw['num_classes']} but COCO has "
            f"{len(categories)} categories: {categories}"
        )

    import torch

    requested_device = (
        raw.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device: {requested_device}")

    for split, images, annotations, _ in summaries:
        dataset = raw[
            "train_dataloader"
            if split == "train"
            else "val_dataloader"
        ]["dataset"]

        print(
            f"{split.title()} data: "
            f"images={images}, "
            f"annotations={annotations}, "
            f"image_dir={dataset['img_folder']}"
        )

    print(
        f"Classes: {len(categories)}; "
        f"COCO category IDs: {category_ids}; "
        f"definitions: {categories}"
    )

    print(
        f"DINOv3 checkpoint: "
        f"{weights_path.resolve()}"
    )

    print(
        "DINOv3 backbone: "
        f"{'trainable' if raw['DINOv3STAs'].get('finetune', True) else 'frozen'}"
    )

    print(
        "Batch size: "
        f"{raw['train_dataloader'].get('total_batch_size', raw['train_dataloader'].get('batch_size'))}"
    )

    print(
        f"Learning rate: {raw['optimizer']['lr']}; "
        f"epochs: {raw['epoches']}"
    )

    model = config.model

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print(
        f"Parameters: "
        f"trainable={trainable:,}; "
        f"frozen={frozen:,}"
    )


def smoke_test(
    config_path: Path,
    overrides: list[str],
) -> None:

    sys.path.insert(0, str(DEIMV2_ROOT))

    import torch

    from engine.core import YAMLConfig, yaml_utils
    from engine.solver.det_engine import evaluate

    cfg = YAMLConfig(
        str(config_path),
        **yaml_utils.parse_cli(overrides),
    )

    device = torch.device(
        cfg.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)

    optimizer = cfg.optimizer

    loader = cfg.train_dataloader
    loader.set_epoch(0)

    samples, targets = next(iter(loader))

    samples = samples.to(device)

    targets = [
        {
            key: value.to(device)
            for key, value in target.items()
        }
        for target in targets
    ]

    print(
        f"Smoke batch: "
        f"images={tuple(samples.shape)}; "
        f"target boxes={[tuple(t['boxes'].shape) for t in targets]}"
    )

    model.train()
    criterion.train()

    outputs = model(
        samples,
        targets=targets,
    )

    loss_dict = criterion(
        outputs,
        targets,
        epoch=0,
        step=0,
        global_step=0,
        epoch_step=1,
    )

    loss = sum(loss_dict.values())

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite training loss: {loss_dict}"
        )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    trainable_with_grad = sum(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_with_grad = sum(
        p.grad is not None
        for p in model.backbone.dinov3.parameters()
        if not p.requires_grad
    )

    if trainable_with_grad == 0:
        raise RuntimeError(
            "No trainable parameter received a gradient."
        )

    if frozen_with_grad:
        raise RuntimeError(
            "Frozen DINOv3 parameters unexpectedly received gradients."
        )

    optimizer.step()

    print(
        f"Smoke train step: "
        f"finite loss={loss.item():.6f}; "
        f"trainable tensors with grad={trainable_with_grad}; "
        f"frozen DINOv3 gradients={frozen_with_grad}"
    )

    evaluator = cfg.evaluator

    one_val_batch = [
        next(iter(cfg.val_dataloader))
    ]

    stats, _ = evaluate(
        model,
        criterion,
        cfg.postprocessor.to(device),
        one_val_batch,
        evaluator,
        device,
    )

    bbox = stats.get(
        "coco_eval_bbox",
        [],
    )

    if bbox:
        print(
            f"Smoke validation: "
            f"AP={bbox[0]:.4f}, "
            f"AP50={bbox[1]:.4f}, "
            f"AP75={bbox[2]:.4f}"
        )


def find_best_checkpoint(output_dir: Path) -> Path:
    """
    Find the checkpoint that should be exported.

    Priority:
        1. best_stg1.pth
        2. best.pth
        3. last.pth
    """

    candidates = [
        output_dir / "best_stg1.pth",
        output_dir / "best.pth",
        output_dir / "last.pth",
    ]

    for checkpoint in candidates:
        if checkpoint.is_file():
            print(
                f"Selected checkpoint for export: {checkpoint}"
            )
            return checkpoint

    # Fallback: search for any .pth file.
    checkpoints = sorted(
        output_dir.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if checkpoints:
        print(
            f"Selected latest checkpoint for export: "
            f"{checkpoints[0]}"
        )
        return checkpoints[0]

    raise FileNotFoundError(
        f"No .pth checkpoint found in {output_dir}"
    )


def export_model_pt(
    config_path: Path,
    overrides: list[str],
) -> Path:
    """
    Convert the selected DEIMv2 checkpoint into model.pt.

    The resulting file contains:
        - model state_dict
        - model configuration
        - class information
        - checkpoint source
    """

    import torch

    sys.path.insert(0, str(DEIMV2_ROOT))

    from engine.core import YAMLConfig, yaml_utils

    config = YAMLConfig(
        str(config_path),
        **yaml_utils.parse_cli(overrides),
    )

    raw = config.yaml_cfg

    output_dir = Path(
        raw["output_dir"]
    ).resolve()

    checkpoint_path = find_best_checkpoint(
        output_dir
    )

    print()
    print("=" * 70)
    print("Exporting DEIMv2 model")
    print("=" * 70)
    print(
        f"Checkpoint: {checkpoint_path}"
    )

    # Build the exact same architecture as during training.
    model = config.model

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Some checkpoints may already be state_dicts.
            state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    # Remove possible DistributedDataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing_keys:
        print(
            "WARNING: Missing keys while loading checkpoint:"
        )
        for key in missing_keys[:20]:
            print(f"  {key}")

    if unexpected_keys:
        print(
            "WARNING: Unexpected keys while loading checkpoint:"
        )
        for key in unexpected_keys[:20]:
            print(f"  {key}")

    if missing_keys:
        raise RuntimeError(
            "Model checkpoint could not be loaded completely. "
            "Refusing to create model.pt."
        )

    model.eval()

    # CPU state_dict is much easier to load later in the frontend.
    state_dict_cpu = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    categories = (
        raw
        .get("train_dataloader", {})
        .get("dataset", {})
        .get("coco", {})
    )

    # Read categories directly from the COCO annotation file.
    ann_file = Path(
        raw["train_dataloader"]["dataset"]["ann_file"]
    )

    category_data = []

    if ann_file.is_file():
        coco = json.loads(
            ann_file.read_text(encoding="utf-8")
        )

        category_data = coco.get(
            "categories",
            [],
        )

    export_data = {
        "model_state_dict": state_dict_cpu,
        "checkpoint": str(
            checkpoint_path
        ),
        "config": raw,
        "categories": category_data,
        "num_classes": raw["num_classes"],
        "format": "DEIMv2-DINOv3",
    }

    model_pt = output_dir / "model.pt"

    torch.save(
        export_data,
        model_pt,
    )

    print(
        f"Model exported successfully:"
        f"\n  {model_pt}"
    )

    print(
        f"Size: "
        f"{model_pt.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print("=" * 70)

    return model_pt


def main() -> None:

    args = parse_args()

    if args.evaluate and args.resume is None:
        raise SystemExit(
            "--evaluate requires --resume PATH "
            "so validation does not score an untrained detector."
        )

    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume}"
        )

    config_path = args.config.resolve()

    require_paths(config_path)

    os.chdir(DEIMV2_ROOT)

    overrides = update_items(args)

    startup_report(
        config_path,
        overrides,
    )

    if args.smoke_test:
        smoke_test(
            config_path,
            overrides,
        )
        return

    command = [
        sys.executable,
        "train.py",
        "--config",
        str(config_path),
        "--seed",
        "0",
    ]

    if args.evaluate:
        command.append("--test-only")

    if overrides:
        command.extend(
            (
                "--update",
                *overrides,
            )
        )

    print(
        "Launching DEIMv2:",
        " ".join(command),
    )

    # Training/evaluation must finish successfully
    # before model.pt is generated.
    subprocess.run(
        command,
        cwd=DEIMV2_ROOT,
        check=True,
    )

    # Do not export a model after --evaluate unless you explicitly
    # want to regenerate model.pt from the evaluated checkpoint.
    if args.evaluate:
        print(
            "Evaluation finished. "
            "Skipping model.pt export."
        )
        return

    # ------------------------------------------------------------
    # NEW:
    # Export trained DEIMv2 + DINOv3 model to model.pt
    # ------------------------------------------------------------
    export_model_pt(
        config_path,
        overrides,
    )


if __name__ == "__main__":
    main()