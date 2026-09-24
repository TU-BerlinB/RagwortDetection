"""
File: src/scripts/import_field_images.py
Usage:
    python src/scripts/import_field_images.py
    python src/scripts/import_field_images.py --target 1000 --countries PL,CZ,DE
    python src/scripts/import_field_images.py --crop 0.5 --save-rejected
    python src/scripts/import_field_images.py --sync-txt
Description:
    Augments the dataset with top-down meadow and grassland negative images (without ragwort)
    sourced from European Commission JRC LUCAS Cover photos (CC BY 4.0).
    Pairs each saved image with an empty YOLO format .txt label file, applies color/sky/flower
    filtering (optionally using zero-shot CLIP), and tracks metadata in metadata/plants_sources.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    import requests
    from PIL import Image, ImageOps
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install via: pip install requests pillow numpy"
    ) from exc

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "plants"
DEFAULT_META_DIR = REPO_ROOT / "metadata"

LUCAS_BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_COVER"
LUCAS_META_ZIP = "tables/lucas_cover_attr.csv.zip"
LICENSE = "CC BY 4.0"
ATTRIBUTION = ("European Commission, Joint Research Centre - LUCAS cover photos "
               "(d'Andrimont et al. 2022, doi:10.5194/essd-14-4463-2022)")
USER_AGENT = "Mozilla/5.0 (compatible; RagwortDetection-plantsextra/1.0)"
IMG_EXT = ".jpg"
FILE_PREFIX = "lucas"
OUR_NAME = re.compile(r"^lucas\d{4}_[A-Z]{2}_\d+$")

DEFAULT_COUNTRIES = "PL,CZ,SK,DE,LT,LV,EE,AT,HU"

SKY_TOP_PART = 0.30
SKY_BLOCK = 16
SKY_STD = 12.0
SKY_LUM = 110.0
MAX_SKY = 0.03
VEG_SAT = 0.08
VEG_HUE = (20.0, 170.0)
MIN_VEG_CLIP = 0.25
MIN_VEG_SIMPLE = 0.50
MAX_TOP_BOTTOM_DIFF = 60
MAX_RED = 0.01
MAX_YELLOW = 0.008
REVIEW_YELLOW = 0.002
CLIP_THRESHOLD = 0.50

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_POSITIVE = [
    "a top-down photo of grass on the ground",
    "a close-up photo looking straight down at meadow vegetation",
    "a photo of dry grass seen from above",
]
CLIP_NEGATIVE = [
    "a landscape photo of a meadow with the horizon",
    "a photo of a field with trees and sky in the background",
    "a photo of a road or a path",
    "a close-up photo of tree leaves or a bush",
    "a photo of a red sign lying on the grass",
    "a photo of a person or an animal",
    "a photo of a building or a fence",
]

SOURCES_FIELDS = ["file", "point_id", "year", "country", "lc1", "lc1_label", "survey_date",
                  "source_url", "license", "attribution", "clip_score", "sky", "veg", "yellow"]
CAND_FIELDS = ["point_id", "year", "country", "lc1", "lc1_label", "survey_date", "url"]

Candidate = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Target directory for images and empty label files (default: data/plants)")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="Separate directory for .txt label files (default: same as images)")
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR,
                        help="Directory for metadata, cache, and logs (default: metadata/)")
    parser.add_argument("--target", type=int, default=1000,
                        help="Target count of images in output directory (default: 1000)")
    parser.add_argument("--countries", default=DEFAULT_COUNTRIES,
                        help=f"Comma-separated country codes or ALL (default: {DEFAULT_COUNTRIES})")
    parser.add_argument("--years", default="2018",
                        help="LUCAS survey years, e.g. 2018 or 2015,2018 (default: 2018)")
    parser.add_argument("--classes", default="E10,E20,E30",
                        help="LUCAS land cover classes; E indicates grasslands (default: E10,E20,E30)")
    parser.add_argument("--crop", type=float, default=0.6,
                        help="Fraction of image center to keep, 0.2-1.0; 1.0 = no crop (default: 0.6)")
    parser.add_argument("--max-side", type=int, default=0,
                        help="Resize longer image edge to this size in pixels (0 = no resize)")
    parser.add_argument("--filter", choices=["auto", "clip", "simple", "none"], default="auto",
                        help="auto = CLIP if torch/transformers available, otherwise simple heuristic")
    parser.add_argument("--clip-model", default=CLIP_MODEL,
                        help=f"Hugging Face CLIP model identifier (default: {CLIP_MODEL})")
    parser.add_argument("--clip-threshold", type=float, default=CLIP_THRESHOLD,
                        help=f"Minimum CLIP confidence for top-down view (default: {CLIP_THRESHOLD})")
    parser.add_argument("--max-yellow", type=float, default=MAX_YELLOW,
                        help=f"Maximum allowed fraction of yellow flowers; 1 = disable (default: {MAX_YELLOW})")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download worker count (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection order (default: 42)")
    parser.add_argument("--save-rejected", action="store_true",
                        help="Save rejected images to metadata/plants_odrzucone for review")
    parser.add_argument("--sync-txt", action="store_true",
                        help="Synchronize missing empty .txt files / remove orphans and exit")
    parser.add_argument("--base-url", default=LUCAS_BASE_URL, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop must be in range 0.2-1.0")
    if args.target < 1:
        parser.error("--target must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


_thread_local = threading.local()


def http() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers["User-Agent"] = USER_AGENT
        _thread_local.session = sess
    return sess


def download_file(url: str, dest: Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            with http().get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r[download] {desc}: {done / 1e6:.0f} / {total / 1e6:.0f} MB",
                                  end="", flush=True)
            print()
            if total and done != total:
                raise IOError(f"Downloaded {done} of {total} bytes")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            print(f"\n[download] error ({attempt}/3): {exc}")
            time.sleep(3 * attempt)
    raise SystemExit(f"Failed to download {url}")


def fetch(cand: Candidate) -> Tuple[Candidate, Optional[bytes], str]:
    last = ""
    for attempt in range(1, 4):
        try:
            resp = http().get(cand["url"], timeout=30)
            if resp.status_code == 404:
                return cand, None, "404"
            resp.raise_for_status()
            return cand, resp.content, ""
        except Exception as exc:
            last = str(exc)[:120]
            time.sleep(1.5 * attempt)
    return cand, None, last


def image_name(cand: Candidate) -> str:
    return f"{FILE_PREFIX}{cand['year']}_{cand['country']}_{cand['point_id']}{IMG_EXT}"


def cand_key(cand: Candidate) -> str:
    return f"{cand['year']}_{cand['point_id']}"


def load_candidates(
    zip_path: Path,
    cache_path: Path,
    years: Set[str],
    countries: Set[str],
    classes: List[str],
    base_url: str,
) -> List[Candidate]:
    if cache_path.exists():
        with open(cache_path, newline="", encoding="utf-8") as fh:
            cands = list(csv.DictReader(fh))
        print(f"[metadata] Loaded {len(cands)} candidates from cache ({cache_path.name})")
    else:
        print("[metadata] Parsing LUCAS table (1.3 GB uncompressed CSV) - this may take 1-3 minutes...")
        csv.field_size_limit(2 ** 31 - 1)
        cands = []
        t0 = time.time()
        with zipfile.ZipFile(zip_path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                header = next(reader)
                ix = {h: i for i, h in enumerate(header)}
                missing = [h for h in ("point_id", "year", "nuts0", "lc1", "file_path_ftp_cover")
                           if h not in ix]
                if missing:
                    raise SystemExit(f"Unexpected LUCAS table format, missing columns: {missing}")
                i_id, i_year, i_cc = ix["point_id"], ix["year"], ix["nuts0"]
                i_lc, i_url = ix["lc1"], ix["file_path_ftp_cover"]
                i_lab, i_date = ix.get("lc1_label"), ix.get("survey_date")
                for n, row in enumerate(reader, 1):
                    if n % 50000 == 0:
                        print(f"\r[metadata] Rows: {n:,}  Candidates: {len(cands):,}  "
                              f"({time.time() - t0:.0f} s)", end="", flush=True)
                    if len(row) != len(header):
                        continue
                    if row[i_year] not in years:
                        continue
                    if countries and row[i_cc] not in countries:
                        continue
                    lc1 = row[i_lc]
                    if not any(lc1.startswith(c) for c in classes):
                        continue
                    url = row[i_url]
                    if not url or url == "NA":
                        continue
                    cands.append({
                        "point_id": row[i_id], "year": row[i_year], "country": row[i_cc],
                        "lc1": lc1,
                        "lc1_label": row[i_lab] if i_lab is not None else "",
                        "survey_date": row[i_date] if i_date is not None else "",
                        "url": url,
                    })
        print()
        seen: Set[str] = set()
        unique: List[Candidate] = []
        for cand in cands:
            key = cand_key(cand)
            if key not in seen:
                seen.add(key)
                unique.append(cand)
        cands = unique
        tmp = cache_path.with_name(cache_path.name + ".part")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CAND_FIELDS)
            writer.writeheader()
            writer.writerows(cands)
        os.replace(tmp, cache_path)
        print(f"[metadata] Found {len(cands)} matching candidate images ({time.time() - t0:.0f} s)")

    if base_url.rstrip("/") != LUCAS_BASE_URL:
        for cand in cands:
            cand["url"] = cand["url"].replace(LUCAS_BASE_URL, base_url.rstrip("/"))
    return cands


def trim_black_borders(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = arr.shape
    rows, cols = arr.mean(axis=1), arr.mean(axis=0)
    t, b, l, r = 0, h - 1, 0, w - 1
    while t < h // 3 and rows[t] < 12:
        t += 1
    while b > 2 * h // 3 and rows[b] < 12:
        b -= 1
    while l < w // 3 and cols[l] < 12:
        l += 1
    while r > 2 * w // 3 and cols[r] < 12:
        r -= 1
    if (t, l, b, r) != (0, 0, h - 1, w - 1):
        img = img.crop((l, t, r + 1, b + 1))
    return img


def hue_deg(R: np.ndarray, G: np.ndarray, B: np.ndarray,
            mx: np.ndarray, mn: np.ndarray) -> np.ndarray:
    d = np.maximum(mx - mn, 1e-6)
    h = np.zeros_like(mx)
    rmax = mx == R
    gmax = (mx == G) & ~rmax
    bmax = ~(rmax | gmax)
    h[rmax] = np.mod((G - B)[rmax] / d[rmax], 6.0)
    h[gmax] = (B - R)[gmax] / d[gmax] + 2.0
    h[bmax] = (R - G)[bmax] / d[bmax] + 4.0
    h = h * 60.0
    h[mx == mn] = 0.0
    return h


def _channels(arr: np.ndarray):
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    hue = hue_deg(R, G, B, mx, mn)
    return R, G, B, lum, mx, sat, hue


def yellow_flower_fraction(img: Image.Image) -> float:
    """Fraction of bright yellow pixels notably brighter than surrounding region."""
    W, H, bw, bh = 480, 360, 24, 18
    arr = np.asarray(img.resize((W, H), Image.BOX), dtype=np.float32)
    R, G, B, lum, mx, sat, hue = _channels(arr)
    small = np.asarray(img.resize((bw, bh), Image.BOX), dtype=np.float32)
    small_lum = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
    local = np.repeat(np.repeat(small_lum, H // bh, axis=0), W // bw, axis=1)
    mask = ((mx >= 190) & (sat >= 0.6) & (B <= 0.45 * mx)
            & (hue >= 40) & (hue <= 62) & (lum - local >= 35))
    return float(mask.mean())


def image_stats(img: Image.Image, crop: float) -> Dict[str, float]:
    """Calculate heuristic metrics: sky presence, vegetation, top/bottom gradient, red board, yellow flowers."""
    W, H = 320, 240
    arr = np.asarray(img.resize((W, H), Image.BOX), dtype=np.float32)
    R, G, B, lum, mx, sat, hue = _channels(arr)

    bs = SKY_BLOCK
    nby, nbx = int(H * SKY_TOP_PART) // bs, W // bs

    def blocks(x: np.ndarray) -> np.ndarray:
        return x[: nby * bs, : nbx * bs].reshape(nby, bs, nbx, bs)

    lb = blocks(lum)
    m, sd = lb.mean(axis=(1, 3)), lb.std(axis=(1, 3))
    mr, mg, mb = (blocks(c).mean(axis=(1, 3)) for c in (R, G, B))
    ms = blocks(sat).mean(axis=(1, 3))
    bluish = (mb >= mr + 5) & (mb >= mg - 5)
    whitish = (ms < 0.18) & (m > 160)
    sky = float(((sd < SKY_STD) & (m > SKY_LUM) & (bluish | whitish)).mean())

    x0, y0 = int(W * (1 - crop) / 2), int(H * (1 - crop) / 2)
    cs = (slice(y0, H - y0), slice(x0, W - x0))
    veg = float(((sat[cs] >= VEG_SAT) & (hue[cs] >= VEG_HUE[0])
                 & (hue[cs] <= VEG_HUE[1]) & (lum[cs] > 20)).mean())

    top = arr[: int(H * 0.25)].reshape(-1, 3).mean(axis=0)
    bottom = arr[H // 2:].reshape(-1, 3).mean(axis=0)
    diff = float(np.linalg.norm(top - bottom))

    red = float(((R > 120) & (R > 1.8 * G) & (R > 1.8 * B)).mean())
    return {"sky": sky, "veg": veg, "diff": diff, "red": red,
            "yellow": yellow_flower_fraction(img)}


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    if frac >= 0.999:
        return img
    w, h = img.size
    cw, ch = int(round(w * frac)), int(round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


class ClipScorer:
    """Zero-shot CLIP scorer estimating top-down viewing angle probability."""

    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[clip] Loading model '{model_name}' on ({self.device})...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.text = self.proc(text=CLIP_POSITIVE + CLIP_NEGATIVE, return_tensors="pt", padding=True)

    def scores(self, images: List[Image.Image]) -> List[float]:
        if not images:
            return []
        with self.torch.no_grad():
            pix = self.proc(images=images, return_tensors="pt")
            out = self.model(
                input_ids=self.text["input_ids"].to(self.device),
                attention_mask=self.text["attention_mask"].to(self.device),
                pixel_values=pix["pixel_values"].to(self.device),
            )
            probs = out.logits_per_image.softmax(dim=-1)
            return probs[:, : len(CLIP_POSITIVE)].sum(dim=-1).float().cpu().tolist()


def make_scorer(mode: str, model_name: str) -> Optional[ClipScorer]:
    if mode in ("simple", "none"):
        return None
    try:
        return ClipScorer(model_name)
    except ImportError:
        if mode == "clip":
            raise SystemExit("Option --filter clip requires: pip install torch transformers")
        print("[filter] Warning: torch or transformers not found. Using simple heuristic filter.")
        return None


def decide(stats: Dict[str, float], clip_score: Optional[float], mode: str,
           clip_threshold: float, max_yellow: float) -> Tuple[bool, str]:
    if mode == "none":
        return True, ""
    if stats["red"] > MAX_RED:
        return False, "red board detected"
    if stats["sky"] > MAX_SKY:
        return False, f"sky detected ({stats['sky']:.2f})"
    if stats["yellow"] > max_yellow:
        return False, f"yellow flowers ({stats['yellow']:.4f})"
    if clip_score is not None:
        if stats["veg"] < MIN_VEG_CLIP:
            return False, f"low vegetation ({stats['veg']:.2f})"
        if clip_score < clip_threshold:
            return False, f"low clip score ({clip_score:.2f})"
        return True, ""
    if stats["veg"] < MIN_VEG_SIMPLE:
        return False, f"low vegetation ({stats['veg']:.2f})"
    if stats["diff"] > MAX_TOP_BOTTOM_DIFF:
        return False, f"angled perspective ({stats['diff']:.0f})"
    return True, ""


def sync_txt(img_dir: Path, lbl_dir: Path) -> Tuple[int, int, int]:
    """Ensure every downloaded image has an empty .txt label and prune orphaned labels."""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stems = {p.stem for p in img_dir.glob("*" + IMG_EXT) if OUR_NAME.match(p.stem)}
    created = removed = 0
    for stem in stems:
        txt = lbl_dir / f"{stem}.txt"
        if not txt.exists():
            txt.touch()
            created += 1
    for txt in lbl_dir.glob(f"{FILE_PREFIX}*.txt"):
        if OUR_NAME.match(txt.stem) and txt.stem not in stems and txt.stat().st_size == 0:
            txt.unlink()
            removed += 1
    return len(stems), created, removed


def read_processed(path: Path) -> Set[str]:
    done: Set[str] = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(row["key"])
    return done


class AppendCsv:
    def __init__(self, path: Path, fields: List[str]):
        new = not path.exists() or path.stat().st_size == 0
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=fields)
        if new:
            self.writer.writeheader()

    def write(self, row: Dict[str, str]) -> None:
        self.writer.writerow(row)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()

    out_dir: Path = args.output_dir
    lbl_dir: Path = args.labels_dir or out_dir
    meta_dir: Path = args.meta_dir
    rej_dir = meta_dir / "plants_odrzucone"
    processed_path = meta_dir / "plants_przetworzone.csv"
    sources_path = meta_dir / "plants_sources.csv"
    review_path = meta_dir / "plants_do_przejrzenia.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    have, created, removed = sync_txt(out_dir, lbl_dir)
    if created or removed:
        print(f"[txt] Added {created} missing, removed {removed} orphaned .txt files")
    if args.sync_txt:
        print(f"[done] Images in {rel(out_dir)}: {have}, each with an empty .txt label file")
        return

    print(f"[start] Target images directory: {rel(out_dir)} | Metadata: {rel(meta_dir)}")
    print(f"[start] Existing images: {have}, Target: {args.target}")
    if have >= args.target:
        print("[done] Target already reached; nothing to do.")
        return

    years = {y.strip() for y in args.years.split(",") if y.strip()}
    countries = (set() if args.countries.strip().upper() == "ALL"
                 else {c.strip().upper() for c in args.countries.split(",") if c.strip()})
    classes = [c.strip().upper() for c in args.classes.split(",") if c.strip()]

    zip_path = meta_dir / "lucas_cover_attr.csv.zip"
    if not zip_path.exists():
        print("[metadata] Downloading LUCAS metadata table (approx 145 MB)...")
        download_file(f"{args.base_url.rstrip('/')}/{LUCAS_META_ZIP}", zip_path, "LUCAS Table")

    tag = "_".join([
        "-".join(sorted(years)),
        "-".join(sorted(countries)) if countries else "ALL",
        "-".join(classes),
    ])
    cache_path = meta_dir / f"lucas_kandydaci_{tag}.csv"
    cands = load_candidates(zip_path, cache_path, years, countries, classes, args.base_url)

    processed = read_processed(processed_path)
    todo = [c for c in cands
            if cand_key(c) not in processed and not (out_dir / image_name(c)).exists()]
    random.Random(args.seed).shuffle(todo)
    print(f"[data] Candidates to check: {len(todo)} (previously evaluated: {len(processed)})")
    if not todo:
        print("[data] No new candidates available. Add countries (--countries) or years (--years 2015,2018).")
        return

    scorer = make_scorer(args.filter, args.clip_model)
    mode = args.filter if args.filter != "auto" else ("clip" if scorer else "simple")
    print(f"[filter] Active mode: {mode}")

    if args.save_rejected:
        rej_dir.mkdir(parents=True, exist_ok=True)

    proc_log = AppendCsv(processed_path, ["key", "status", "info"])
    sources = AppendCsv(sources_path, SOURCES_FIELDS)

    accepted = have
    checked = rejected = errors = to_review = 0
    t0 = time.time()
    batch_size = max(8, args.workers * 4)
    pos = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while accepted < args.target and pos < len(todo):
                batch = todo[pos: pos + batch_size]
                pos += batch_size

                items: List[list] = []
                for cand, data, err in pool.map(fetch, batch):
                    if err:
                        errors += 1
                        proc_log.write({"key": cand_key(cand), "status": "error", "info": err})
                        continue
                    try:
                        img = Image.open(io.BytesIO(data))
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        img = trim_black_borders(img)
                        if min(img.size) < 200:
                            raise ValueError(f"Image too small {img.size}")
                    except Exception as exc:
                        errors += 1
                        proc_log.write({"key": cand_key(cand), "status": "error",
                                        "info": f"image: {str(exc)[:80]}"})
                        continue
                    items.append([cand, img, image_stats(img, args.crop), None])

                if scorer:
                    need = [it for it in items
                            if decide(it[2], 1.0, "clip", 0.0, args.max_yellow)[0]]
                    for i in range(0, len(need), 16):
                        chunk = need[i: i + 16]
                        for it, score in zip(chunk, scorer.scores([it[1] for it in chunk])):
                            it[3] = score

                for cand, img, st, clip_score in items:
                    if accepted >= args.target:
                        break
                    checked += 1
                    use_clip = clip_score if mode == "clip" else None
                    if mode == "clip" and clip_score is None:
                        use_clip = -1.0
                    ok, why = decide(st, use_clip, mode, args.clip_threshold, args.max_yellow)
                    name = image_name(cand)
                    if ok:
                        out_img = center_crop(img, args.crop)
                        if args.max_side and max(out_img.size) > args.max_side:
                            out_img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
                        tmp = out_dir / (name + ".part")
                        out_img.save(tmp, "JPEG", quality=95)
                        os.replace(tmp, out_dir / name)
                        (lbl_dir / f"{Path(name).stem}.txt").touch()
                        accepted += 1
                        if st["yellow"] > REVIEW_YELLOW:
                            to_review += 1
                            with open(review_path, "a", encoding="utf-8") as fh:
                                fh.write(f"{name}\tyellow={st['yellow']:.4f}\n")
                        sources.write({
                            "file": name, "point_id": cand["point_id"], "year": cand["year"],
                            "country": cand["country"], "lc1": cand["lc1"],
                            "lc1_label": cand["lc1_label"], "survey_date": cand["survey_date"],
                            "source_url": cand["url"], "license": LICENSE,
                            "attribution": ATTRIBUTION,
                            "clip_score": "" if clip_score is None else f"{clip_score:.3f}",
                            "sky": f"{st['sky']:.3f}", "veg": f"{st['veg']:.3f}",
                            "yellow": f"{st['yellow']:.4f}",
                        })
                        proc_log.write({"key": cand_key(cand), "status": "ok", "info": name})
                    else:
                        rejected += 1
                        proc_log.write({"key": cand_key(cand), "status": "rejected", "info": why})
                        if args.save_rejected:
                            small = img.copy()
                            small.thumbnail((800, 800))
                            small.save(rej_dir / name, "JPEG", quality=85)

                elapsed = (time.time() - t0) / 60
                print(f"\r[progress] Saved: {accepted}/{args.target} | Checked: {checked} | "
                      f"Rejected: {rejected} | Errors: {errors} | {elapsed:.1f} min   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stop] Interrupted by user.")
    finally:
        print()
        proc_log.close()
        sources.close()

    have, _, _ = sync_txt(out_dir, lbl_dir)
    print(f"[done] Images in {rel(out_dir)}: {have} (each with empty .txt label)")
    print(f"[done] Image provenance recorded in: {rel(sources_path)}")
    if to_review:
        print(f"[notice] {to_review} images flagged with subtle yellow flowers; review at: {rel(review_path)}")


if __name__ == "__main__":
    main()
