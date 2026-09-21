"""
Generator zdjec syntetycznych metoda copy-paste: wycieta roslina starca ladowana jest
na losowe tlo laki, a ramka YOLO powstaje automatycznie z maski wklejki - wiec jest
idealnie dokladna i nie wymaga adnotacji recznej.

Tlo i pierwszy plan pochodza z tego samego zbioru dronowego (11 m, 3,18 mm/piksel),
wiec skala obu warstw jest zgodna - to warunek, zeby syntetyk nie wygladal absurdalnie.

KOMENDY - z /workspace:

# 1. 2000 zdjec 512x512, po 1-4 rosliny na zdjecie:
python src/scripts/make_synthetic.py

# 2. wiecej, wieksze i gesciej:
python src/scripts/make_synthetic.py --count 5000 --size 640 --max-obj 6

# 3. podglad z narysowanymi ramkami, do sprawdzenia okiem:
python src/scripts/make_synthetic.py --count 24 --preview

WEJSCIE:  data/meadow_negatives/images/   tla
          data/ragwort_cutouts/           wyciete rosliny
WYJSCIE:  data/synthetic/images/*.jpg + labels/*.txt   gotowe do YOLO
          data/synthetic/index.csv                     z czego powstalo kazde zdjecie
          data/synthetic/preview/                      tylko z --preview

UWAGA: syntetyki nie zastepuja prawdziwych zdjec. Trzymaj je jako dodatek do zbioru
treningowego (sensownie do polowy calosci), a zbior walidacyjny i testowy buduj
WYLACZNIE z prawdziwych zdjec - inaczej metryki beda klamac.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise SystemExit(f"Blad: nie ma katalogu {folder}. Najpierw: python src/scripts/get_meadow_negatives.py")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"Blad: w {folder} nie ma zdjec.")
    return files


def build_background(bg_files: List[Path], size: int, rng: random.Random) -> Tuple["Image.Image", List[str]]:
    """Tlo o zadanym rozmiarze. Male zdjecia sklejane sa w mozaike zamiast rozciagane."""
    from PIL import Image

    used: List[str] = []
    canvas = Image.new("RGB", (size, size))

    first = Image.open(rng.choice(bg_files)).convert("RGB")
    if min(first.size) >= size:
        scale = size / min(first.size)
        new = (max(size, int(first.width * scale)), max(size, int(first.height * scale)))
        first = first.resize(new, Image.LANCZOS)
        x = rng.randint(0, first.width - size)
        y = rng.randint(0, first.height - size)
        canvas.paste(first.crop((x, y, x + size, y + size)), (0, 0))
        return canvas, [first.filename if hasattr(first, "filename") else "bg"]

    tile = max(96, min(size, max(first.size)))
    for ty in range(0, size, tile):
        for tx in range(0, size, tile):
            path = rng.choice(bg_files)
            im = Image.open(path).convert("RGB")
            scale = tile / min(im.size)
            im = im.resize((max(tile, int(im.width * scale)), max(tile, int(im.height * scale))), Image.LANCZOS)
            im = im.crop((0, 0, tile, tile))
            canvas.paste(im, (tx, ty))
            used.append(path.name)
    return canvas, used


def make_cutout(path: Path, target_px: int, rng: random.Random):
    """Zwraca (RGB, alfa) rosliny: elipsa z rozmyta krawedzia, obrocona losowo."""
    from PIL import Image, ImageDraw, ImageFilter

    im = Image.open(path).convert("RGB")
    scale = target_px / max(im.size)
    im = im.resize((max(8, int(im.width * scale)), max(8, int(im.height * scale))), Image.LANCZOS)

    w, h = im.size
    inset_x = int(w * rng.uniform(0.04, 0.12))
    inset_y = int(h * rng.uniform(0.04, 0.12))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([inset_x, inset_y, w - inset_x - 1, h - inset_y - 1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(w, h) * 0.05)))

    angle = rng.uniform(0, 360)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)

    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return im, mask


def harmonize(fg, bg_patch, strength: float):
    """Czesciowe dopasowanie sredniej i kontrastu wklejki do tla, zeby nie odcinala sie kolorem."""
    if strength <= 0:
        return fg
    from PIL import Image

    a = np.asarray(fg, dtype=np.float32)
    b = np.asarray(bg_patch.resize(fg.size), dtype=np.float32)
    for c in range(3):
        fa, fs = a[..., c].mean(), a[..., c].std() or 1.0
        ba, bs = b[..., c].mean(), b[..., c].std() or 1.0
        adjusted = (a[..., c] - fa) * (bs / fs) + ba
        a[..., c] = a[..., c] * (1 - strength) + adjusted * strength
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def mask_bbox(mask, threshold: int = 24):
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr >= threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-dir", type=Path, default=REPO_ROOT / "data" / "meadow_negatives" / "images")
    ap.add_argument("--fg-dir", type=Path, default=REPO_ROOT / "data" / "ragwort_cutouts")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synthetic")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--min-obj", type=int, default=1)
    ap.add_argument("--max-obj", type=int, default=4)
    ap.add_argument("--min-scale", type=float, default=0.10, help="udzial dluzszego boku rosliny w kadrze")
    ap.add_argument("--max-scale", type=float, default=0.30)
    ap.add_argument("--harmonize", type=float, default=0.45, help="0 = brak dopasowania koloru, 1 = pelne")
    ap.add_argument("--max-iou", type=float, default=0.25, help="ile rosliny moga na siebie nachodzic")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", action="store_true", help="zapisz tez wersje z narysowanymi ramkami")
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    rng = random.Random(args.seed)
    bg_files = list_images(args.bg_dir)
    fg_files = list_images(args.fg_dir)
    print(f"[dane] tla: {len(bg_files)}, rosliny: {len(fg_files)}")

    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out / "preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    boxes_total = 0

    for n in range(1, args.count + 1):
        canvas, bg_used = build_background(bg_files, args.size, rng)
        placed: List[Tuple[int, int, int, int]] = []
        used_fg: List[str] = []

        for _ in range(rng.randint(args.min_obj, args.max_obj)):
            fg_path = rng.choice(fg_files)
            target_px = int(args.size * rng.uniform(args.min_scale, args.max_scale))
            fg, mask = make_cutout(fg_path, target_px, rng)

            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bw < 8 or bh < 8 or bw >= args.size or bh >= args.size:
                continue

            for _ in range(20):
                px = rng.randint(-bbox[0], args.size - bbox[2])
                py = rng.randint(-bbox[1], args.size - bbox[3])
                abs_box = (px + bbox[0], py + bbox[1], px + bbox[2], py + bbox[3])
                if all(iou(abs_box, prev) <= args.max_iou for prev in placed):
                    break
            else:
                continue

            patch_box = (max(0, px), max(0, py), min(args.size, px + fg.width), min(args.size, py + fg.height))
            bg_patch = canvas.crop(patch_box)
            fg = harmonize(fg, bg_patch, args.harmonize)

            canvas.paste(fg, (px, py), mask)
            placed.append(abs_box)
            used_fg.append(fg_path.name)

        if not placed:
            continue

        stem = f"syn_{n:06d}"
        quality = rng.randint(80, 95)
        canvas.save(images_dir / f"{stem}.jpg", "JPEG", quality=quality)

        lines = []
        for (x1, y1, x2, y2) in placed:
            cx = (x1 + x2) / 2 / args.size
            cy = (y1 + y2) / 2 / args.size
            w = (x2 - x1) / args.size
            h = (y2 - y1) / args.size
            lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        boxes_total += len(lines)

        if args.preview:
            vis = canvas.copy()
            d = ImageDraw.Draw(vis)
            for b in placed:
                d.rectangle(b, outline=(255, 0, 255), width=2)
            vis.save(preview_dir / f"{stem}.jpg", "JPEG", quality=90)

        index_rows.append({"plik": f"{stem}.jpg", "obiektow": len(placed),
                           "tla": ";".join(dict.fromkeys(bg_used))[:200], "rosliny": ";".join(used_fg)})

        if n % 250 == 0:
            print(f"   {n}/{args.count}", flush=True)

    with open(args.out / "index.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["plik", "obiektow", "tla", "rosliny"])
        w.writeheader()
        w.writerows(index_rows)

    print(f"\n[gotowe] zdjec: {len(index_rows)}, ramek: {boxes_total} "
          f"(srednio {boxes_total / max(1, len(index_rows)):.2f} na zdjecie)")
    print(f"[gotowe] {images_dir}")
    print(f"[gotowe] {labels_dir}")
    if args.preview:
        print(f"[gotowe] podglad z ramkami: {preview_dir}")
    print("\nPamietaj: walidacja i test wylacznie na prawdziwych zdjeciach.")


if __name__ == "__main__":
    main()
