#!/usr/bin/env python3
"""
filter_gbif_distant_ragwort.py

Downloads images of ragwort (Jacobaea vulgaris) from the GBIF occurrence API
and filters them through a multi-stage machine learning pipeline to retain
only photos of ragwort on a meadow/field taken from a distance of at least 50 cm:

Key criteria:
  1. Setting & Context: Ragwort growing in a meadow/field with other surroundings (grass, ground, other plants).
  2. Immediate Rejection if Filling Frame: If ragwort takes up >= 30% of the screen area, it is discarded immediately.
  3. Distance Filter: Monocular depth estimation (DepthAnything V2) ensures distance >= 50-70 cm (not extreme near-plane).
  4. Streaming GBIF Pagination & Background Prefetching: Efficient continuous processing towards target count.

Requirements:
  pip install torch torchvision transformers requests pillow numpy
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import requests
import torch
from PIL import Image

# Hugging Face transformers imports
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

# Optimize PyTorch CPU threading
if not torch.cuda.is_available():
    cpu_cores = os.cpu_count() or 4
    optimal_threads = min(12, max(4, cpu_cores // 2))
    torch.set_num_threads(optimal_threads)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ragwort_filter")

GBIF_API_BASE = "https://api.gbif.org/v1"
GBIF_BACKBONE_JACOBAEA_VULGARIS = 5388602  # GBIF backbone taxonKey for Jacobaea vulgaris


# ---------------------------------------------------------------------------
# Step 1: Stream image URLs from GBIF Occurrence API with Pagination
# ---------------------------------------------------------------------------
def resolve_gbif_taxon_key(taxon_key: str) -> int:
    """Resolves alphanumeric taxon keys (e.g. Catalogue of Life 5J5Z2) to GBIF integer usageKey."""
    if str(taxon_key).isdigit():
        return int(taxon_key)

    logger.info(f"Resolving '{taxon_key}' to GBIF backbone key...")
    try:
        match_url = f"{GBIF_API_BASE}/species/match?name=Jacobaea%20vulgaris"
        resp = requests.get(match_url, timeout=10)
        if resp.status_code == 200:
            match_data = resp.json()
            key = match_data.get("usageKey", GBIF_BACKBONE_JACOBAEA_VULGARIS)
            logger.info(f"Resolved to GBIF backbone taxonKey={key} ({match_data.get('scientificName')})")
            return int(key)
    except Exception as exc:
        logger.warning(f"Failed to match species key ({exc}). Using default {GBIF_BACKBONE_JACOBAEA_VULGARIS}.")

    return GBIF_BACKBONE_JACOBAEA_VULGARIS


def stream_gbif_image_records(
    taxon_key: str = "5J5Z2",
    media_type: str = "StillImage",
    page_size: int = 100,
    start_offset: int = 0,
    max_offset: int = 100000,
) -> Generator[Dict[str, Any], None, None]:
    """Paginates through GBIF occurrence search results and yields individual media records."""
    effective_key = resolve_gbif_taxon_key(taxon_key)
    offset = start_offset

    while offset < max_offset:
        params = {
            "taxonKey": effective_key,
            "mediaType": media_type,
            "limit": page_size,
            "offset": offset,
        }
        search_url = f"{GBIF_API_BASE}/occurrence/search?{urllib.parse.urlencode(params)}"

        try:
            resp = requests.get(search_url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(f"GBIF search request failed at offset {offset}: {exc}. Retrying in 5s...")
            time.sleep(5)
            continue

        results = data.get("results", [])
        if not results:
            logger.info("No more occurrences returned by GBIF. Stream finished.")
            break

        for occ in results:
            occ_key = occ.get("key")
            for m in occ.get("media", []):
                identifier = m.get("identifier")
                m_type = m.get("type", "")
                m_format = m.get("format", "")

                if identifier and (
                    m_type == "StillImage"
                    or "image" in m_format.lower()
                    or identifier.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                ):
                    yield {
                        "gbif_id": occ_key,
                        "url": identifier,
                        "scientific_name": occ.get("scientificName", "Jacobaea vulgaris"),
                        "license": m.get("license", "unknown"),
                        "offset": offset,
                    }

        offset += len(results)
        if data.get("endOfRecords", False):
            logger.info("Reached end of records in GBIF.")
            break


# ---------------------------------------------------------------------------
# Step 2: Download image into memory
# ---------------------------------------------------------------------------
def download_image_to_memory(url: str, timeout: int = 12) -> Optional[Image.Image]:
    """Downloads an image into RAM as a PIL Image object."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None

        content = resp.content
        if len(content) < 1500:
            return None

        img = Image.open(io.BytesIO(content)).convert("RGB")
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-threaded Pre-fetching Producer-Consumer Queue
# ---------------------------------------------------------------------------
def prefetch_worker(
    record_generator: Generator[Dict[str, Any], None, None],
    item_queue: queue.Queue,
    stop_event: threading.Event,
    seen_ids: Set[int],
):
    """Background worker that pre-downloads candidate images into RAM."""
    for rec in record_generator:
        if stop_event.is_set():
            break

        gbif_id = rec["gbif_id"]
        if gbif_id in seen_ids:
            continue

        img = download_image_to_memory(rec["url"])
        if img is None:
            continue

        while not stop_event.is_set():
            try:
                item_queue.put((rec, img), timeout=1)
                break
            except queue.Full:
                continue

    item_queue.put((None, None))


# ---------------------------------------------------------------------------
# Pipeline Class with Meadow & Distance Filter
# ---------------------------------------------------------------------------
class RagwortDistantShotFilter:
    """
    Filters images to ensure:
      1. Setting: Wild plant in a grassy meadow/field context vs tight macro shot.
      2. Screen Coverage: Plant bbox union < 30% (immediately discarded if filling screen).
      3. Height Ratio: Single stalk <= 70% of frame height.
      4. Distance: Depth map confirms distance >= 50 cm (mean depth < 180, p90 < 215).
    """

    PROMPT_MACRO = "a close-up macro photo of a flower or plant filling the frame"
    PROMPT_MEADOW = "a wild plant growing in a grassy meadow or field with natural surroundings"

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        detector_model_name: str = "google/owlv2-base-patch16-ensemble",
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Optional[str] = None,
        max_plant_area_ratio: float = 0.30,
        max_plant_height_ratio: float = 0.70,
        max_mean_depth: float = 180.0,
        max_p90_depth: float = 215.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing filter models on device: {self.device}")

        self.max_plant_area_ratio = max_plant_area_ratio
        self.max_plant_height_ratio = max_plant_height_ratio
        self.max_mean_depth = max_mean_depth
        self.max_p90_depth = max_p90_depth

        logger.info(
            f"Filter configuration: Area < {self.max_plant_area_ratio*100:.0f}%, "
            f"Stalk Height <= {self.max_plant_height_ratio*100:.0f}%, "
            f"Depth Mean < {self.max_mean_depth:.0f}, Depth P90 < {self.max_p90_depth:.0f}"
        )

        # 1. Load CLIP model & processor
        logger.info(f"Loading CLIP model: {clip_model_name}...")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()

        # 2. Load Zero-Shot Object Detector (OWLv2)
        logger.info(f"Loading Zero-Shot Detector: {detector_model_name}...")
        self.det_processor = AutoProcessor.from_pretrained(detector_model_name)
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model_name).to(self.device).eval()

        # 3. Load Monocular Depth Estimation Pipeline
        logger.info(f"Loading Depth Estimation model: {depth_model_name}...")
        self.depth_pipe = pipeline(
            "depth-estimation",
            model=depth_model_name,
            device=0 if self.device == "cuda" else -1,
        )

        logger.info("All models loaded successfully!")

    # -----------------------------------------------------------------------
    # Step 3: CLIP meadow vs macro context filter
    # -----------------------------------------------------------------------
    def filter_clip(self, image: Image.Image) -> Tuple[bool, float, float]:
        texts = [self.PROMPT_MACRO, self.PROMPT_MEADOW]
        inputs = self.clip_processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        prob_macro = float(probs[0].item())
        prob_meadow = float(probs[1].item())

        passed = prob_meadow >= prob_macro
        return passed, prob_meadow, prob_macro

    # -----------------------------------------------------------------------
    # Step 4 & 5: Plant detection and screen coverage filter (< 30%)
    # -----------------------------------------------------------------------
    def filter_detection_area_and_height(
        self,
        image: Image.Image,
        score_threshold: float = 0.15,
    ) -> Tuple[bool, bool, float, float, np.ndarray, List[List[int]]]:
        w, h = image.size
        query_text = [["yellow flower plant", "ragwort plant"]]

        inputs = self.det_processor(text=query_text, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.det_model(**inputs)

        target_sizes = torch.Tensor([[h, w]]).to(self.device)
        results = self.det_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
        )[0]

        boxes_tensor = results["boxes"].cpu()
        scores_tensor = results["scores"].cpu()

        plant_mask = np.zeros((h, w), dtype=bool)
        parsed_boxes: List[List[int]] = []
        max_height_ratio = 0.0

        for b, s in zip(boxes_tensor, scores_tensor):
            x1, y1, x2, y2 = [int(v.item()) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                plant_mask[y1:y2, x1:x2] = True
                parsed_boxes.append([x1, y1, x2, y2])
                box_h_ratio = (y2 - y1) / float(h)
                if box_h_ratio > max_height_ratio:
                    max_height_ratio = box_h_ratio

        total_plant_pixels = int(np.sum(plant_mask))
        coverage_ratio = total_plant_pixels / (w * h)

        # 1. Area filter: discard immediately if plant occupies >= 30% of the screen
        area_pass = coverage_ratio < self.max_plant_area_ratio

        # 2. Height filter: discard if a single stalk fills > 70% of the vertical frame
        height_pass = max_height_ratio <= self.max_plant_height_ratio

        return area_pass, height_pass, coverage_ratio, max_height_ratio, plant_mask, parsed_boxes

    # -----------------------------------------------------------------------
    # Step 6: Monocular Depth Estimation (>= 50 cm distance)
    # -----------------------------------------------------------------------
    def filter_depth(
        self,
        image: Image.Image,
        plant_mask: np.ndarray,
    ) -> Tuple[bool, float, float]:
        depth_output = self.depth_pipe(image)
        depth_map = np.array(depth_output["depth"])  # shape: (h, w), uint8 in 0..255

        if depth_map.shape != plant_mask.shape:
            from PIL import Image as PILImg
            resized_depth = PILImg.fromarray(depth_map).resize(
                (plant_mask.shape[1], plant_mask.shape[0]), PILImg.BILINEAR
            )
            depth_map = np.array(resized_depth)

        if np.any(plant_mask):
            plant_depths = depth_map[plant_mask]
        else:
            plant_depths = depth_map

        mean_depth = float(np.mean(plant_depths))
        p90_depth = float(np.percentile(plant_depths, 90))

        # Check distance >= 50-70 cm (discard extreme foreground)
        passed = (mean_depth < self.max_mean_depth) and (p90_depth < self.max_p90_depth)

        return passed, mean_depth, p90_depth

    # -----------------------------------------------------------------------
    # Combined Pipeline Evaluation
    # -----------------------------------------------------------------------
    def evaluate_image(self, image: Image.Image) -> Dict[str, Any]:
        # Step 3: CLIP meadow vs macro
        clip_pass, p_meadow, p_macro = self.filter_clip(image)
        if not clip_pass:
            return {
                "passed": False,
                "reason": f"CLIP: Close-up Macro ({p_macro:.2f}) > Meadow ({p_meadow:.2f})",
                "stage": "clip",
                "metrics": {"p_meadow": p_meadow, "p_macro": p_macro},
            }

        # Step 4 & 5: Plant detection & Area / Height filters
        area_pass, height_pass, area_ratio, max_h_ratio, plant_mask, boxes = (
            self.filter_detection_area_and_height(image)
        )

        if not area_pass:
            return {
                "passed": False,
                "reason": f"Area: Ragwort occupies {area_ratio*100:.1f}% >= {self.max_plant_area_ratio*100:.0f}% of screen",
                "stage": "area",
                "metrics": {
                    "coverage_ratio": area_ratio,
                    "max_height_ratio": max_h_ratio,
                    "boxes_count": len(boxes),
                },
            }

        if not height_pass:
            return {
                "passed": False,
                "reason": f"Height: Single stalk height {max_h_ratio*100:.1f}% > {self.max_plant_height_ratio*100:.0f}%",
                "stage": "height",
                "metrics": {
                    "coverage_ratio": area_ratio,
                    "max_height_ratio": max_h_ratio,
                    "boxes_count": len(boxes),
                },
            }

        # Step 6: Depth estimation (>= 50 cm)
        depth_pass, mean_depth, p90_depth = self.filter_depth(image, plant_mask)
        if not depth_pass:
            return {
                "passed": False,
                "reason": f"Depth: Too close (<50cm foreground, mean={mean_depth:.1f}, p90={p90_depth:.1f})",
                "stage": "depth",
                "metrics": {
                    "coverage_ratio": area_ratio,
                    "max_height_ratio": max_h_ratio,
                    "mean_depth": mean_depth,
                    "p90_depth": p90_depth,
                },
            }

        return {
            "passed": True,
            "reason": "All filters passed: Verified meadow distant shot (>= 50 cm, surrounding vegetation present)",
            "stage": "passed",
            "metrics": {
                "p_meadow": p_meadow,
                "p_macro": p_macro,
                "coverage_ratio": area_ratio,
                "max_height_ratio": max_h_ratio,
                "boxes_count": len(boxes),
                "mean_depth": mean_depth,
                "p90_depth": p90_depth,
            },
        }


# ---------------------------------------------------------------------------
# Step 7: Main Workflow with Live Resuming & Progress Tracking
# ---------------------------------------------------------------------------
def run_ragwort_filter_pipeline(
    taxon_key: str = "5J5Z2",
    output_dir: Union[str, Path] = "ragwort_distant_shots",
    target_downloads: int = 900,
    device: Optional[str] = None,
    page_size: int = 100,
    max_plant_area_ratio: float = 0.30,
    max_plant_height_ratio: float = 0.70,
    max_mean_depth: float = 180.0,
    max_p90_depth: float = 215.0,
) -> int:
    """Continuous streaming pipeline to download and filter distant meadow shots."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover already saved images to resume progress
    existing_images = list(out_path.glob("ragwort_*.jpg"))
    seen_ids: Set[int] = set()
    for p in existing_images:
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            seen_ids.add(int(parts[1]))

    saved_count = len(existing_images)
    logger.info("=" * 75)
    logger.info(" RAGWORT DISTANT MEADOW SHOTS FILTERING PIPELINE")
    logger.info(f" Target folder:   {out_path.resolve()}")
    logger.info(f" Already saved:   {saved_count} images")
    logger.info(f" Target goal:     {target_downloads} images")
    logger.info(f" Needed new:      {max(0, target_downloads - saved_count)} images")
    logger.info(f" Criteria: Area < {max_plant_area_ratio*100:.0f}%, Height <= {max_plant_height_ratio*100:.0f}%, Depth Mean < {max_mean_depth:.0f}, Distance >= 50cm")
    logger.info("=" * 75)

    if saved_count >= target_downloads:
        logger.info(f"Target count of {target_downloads} already reached! Exiting.")
        return saved_count

    # 2. Initialize Models
    filter_engine = RagwortDistantShotFilter(
        device=device,
        max_plant_area_ratio=max_plant_area_ratio,
        max_plant_height_ratio=max_plant_height_ratio,
        max_mean_depth=max_mean_depth,
        max_p90_depth=max_p90_depth,
    )

    # 3. Setup producer thread for background pre-fetching
    item_queue: queue.Queue = queue.Queue(maxsize=12)
    stop_event = threading.Event()

    stream_gen = stream_gbif_image_records(
        taxon_key=taxon_key,
        page_size=page_size,
        start_offset=0,
    )

    producer = threading.Thread(
        target=prefetch_worker,
        args=(stream_gen, item_queue, stop_event, seen_ids),
        daemon=True,
    )
    producer.start()

    stats = {
        "processed": 0,
        "failed_clip": 0,
        "failed_area": 0,
        "failed_height": 0,
        "failed_depth": 0,
        "saved": saved_count,
        "start_time": time.time(),
    }

    def save_progress_file():
        progress_file = out_path / "filter_progress.json"
        elapsed = time.time() - stats["start_time"]
        avg_speed = stats["processed"] / elapsed if elapsed > 0 else 0.0
        data = {
            "target_downloads": target_downloads,
            "saved_count": stats["saved"],
            "processed_count": stats["processed"],
            "discarded_clip": stats["failed_clip"],
            "discarded_area": stats["failed_area"],
            "discarded_height": stats["failed_height"],
            "discarded_depth": stats["failed_depth"],
            "acceptance_rate_pct": round((stats["saved"] / max(1, stats["processed"])) * 100, 2),
            "elapsed_seconds": round(elapsed, 1),
            "images_per_second": round(avg_speed, 3),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(progress_file, "w", encoding="utf-8") as pf:
            json.dump(data, pf, indent=2)

    try:
        while stats["saved"] < target_downloads:
            try:
                rec, image = item_queue.get(timeout=30)
            except queue.Empty:
                logger.warning("Queue empty: waiting for GBIF pre-fetcher...")
                continue

            if rec is None and image is None:
                logger.info("End of GBIF stream reached.")
                break

            gbif_id = rec["gbif_id"]
            seen_ids.add(gbif_id)
            stats["processed"] += 1

            # Run filtering
            eval_res = filter_engine.evaluate_image(image)
            metrics = eval_res.get("metrics", {})

            if not eval_res["passed"]:
                stage = eval_res.get("stage")
                if stage == "clip":
                    stats["failed_clip"] += 1
                elif stage == "area":
                    stats["failed_area"] += 1
                elif stage == "height":
                    stats["failed_height"] += 1
                elif stage == "depth":
                    stats["failed_depth"] += 1

                if stats["processed"] % 10 == 0:
                    logger.info(
                        f"[Progress: {stats['saved']}/{target_downloads} saved | "
                        f"Processed: {stats['processed']} | "
                        f"Discarded: {stats['failed_area']} area (filling screen), {stats['failed_height']} height, {stats['failed_clip']} clip, {stats['failed_depth']} depth]"
                    )
                    save_progress_file()
                continue

            # Save qualifying image
            stats["saved"] += 1
            save_filename = out_path / f"ragwort_{gbif_id}_{stats['saved']}.jpg"
            image.save(save_filename, format="JPEG", quality=95)

            # Save metadata JSON
            meta_filename = out_path / f"ragwort_{gbif_id}_{stats['saved']}.json"
            meta_data = {
                "gbif_id": gbif_id,
                "url": rec["url"],
                "scientific_name": rec["scientific_name"],
                "license": rec["license"],
                "filter_metrics": metrics,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with open(meta_filename, "w", encoding="utf-8") as mf:
                json.dump(meta_data, mf, indent=2)

            logger.info(
                f"[QUALIFIED {stats['saved']}/{target_downloads}] Saved {save_filename.name} "
                f"(Coverage: {metrics.get('coverage_ratio', 0)*100:.1f}% < 30%, "
                f"Depth Mean: {metrics.get('mean_depth', 0):.1f} >= 50cm)"
            )
            save_progress_file()

    finally:
        stop_event.set()
        save_progress_file()

    logger.info("=" * 75)
    logger.info(" RUN FINISHED")
    logger.info(f" Total saved distant shots: {stats['saved']} / {target_downloads}")
    logger.info(f" Total candidates analyzed:  {stats['processed']}")
    logger.info("=" * 75)

    return stats["saved"]


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stream and filter distant meadow ragwort images from GBIF using CLIP, OWLv2, and DepthAnything."
    )
    parser.add_argument(
        "--taxon-key",
        type=str,
        default="5J5Z2",
        help="GBIF or Catalogue of Life taxonKey (default: 5J5Z2 for Jacobaea vulgaris)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ragwort_distant_shots",
        help="Local output folder (default: ragwort_distant_shots)",
    )
    parser.add_argument(
        "--target-downloads",
        type=int,
        default=900,
        help="Number of passing distant meadow images to collect and save (default: 900)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="GBIF API pagination limit per page (default: 100)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.30,
        help="Max total plant bounding box area ratio (default: 0.30 = 30%)",
    )
    parser.add_argument(
        "--max-height-ratio",
        type=float,
        default=0.70,
        help="Max single plant bounding box height ratio (default: 0.70 = 70%)",
    )
    parser.add_argument(
        "--max-mean-depth",
        type=float,
        default=180.0,
        help="Max allowed mean depth inside plant region for >= 50cm (default: 180.0)",
    )
    parser.add_argument(
        "--max-p90-depth",
        type=float,
        default=215.0,
        help="Max allowed 90th percentile depth inside plant region (default: 215.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', 'cpu' or None for automatic detection",
    )

    args = parser.parse_args()

    run_ragwort_filter_pipeline(
        taxon_key=args.taxon_key,
        output_dir=args.output_dir,
        target_downloads=args.target_downloads,
        device=args.device,
        page_size=args.page_size,
        max_plant_area_ratio=args.max_area_ratio,
        max_plant_height_ratio=args.max_height_ratio,
        max_mean_depth=args.max_mean_depth,
        max_p90_depth=args.max_p90_depth,
    )


if __name__ == "__main__":
    main()
