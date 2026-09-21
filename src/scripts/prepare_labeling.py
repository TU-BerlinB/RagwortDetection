"""
Przygotowanie folderu do adnotacji w labelImg: pobiera zdjecia z wybranego CSV
i zapisuje je pod czytelnymi nazwami, w kolejnosci rankingu.

Dwa tryby:
  domyslny  - pobiera zdjecia do labeling/images/
  --cleanup - po adnotacji przenosi nieopisane zdjecia do labeling/rejected/

KOMENDY - uruchamiaj tam, gdzie jest internet (kontener), z /workspace:

# 1. pobranie calej puli 3000 (ok. 2-4 min, ~1 GB):
python src/scripts/prepare_labeling.py

# 2. tylko pierwsze 500 wg rankingu podobienstwa:
python src/scripts/prepare_labeling.py --limit 500

# 3. inny plik zrodlowy, np. finalowe 200:
python src/scripts/prepare_labeling.py --csv afterfacilitylocation/seed_selected_200.csv

# 4. PO adnotacji: nieopisane zdjecia laduja w labeling/rejected/
python src/scripts/prepare_labeling.py --cleanup

USTAWIENIA W labelImg:
  Open Dir        -> labeling/images
  Change Save Dir -> labeling/labels
  format          -> YOLO (przycisk na lewym pasku, domyslnie PascalVOC)
  View            -> Auto Save Mode
  skroty: W ramka, D nastepne, A poprzednie
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_CSV = REPO_ROOT / "afterfacilitylocation" / "seed_pool_3000.csv"
DEFAULT_OUT = REPO_ROOT / "labeling"
USER_AGENT = "RagwortDetection/1.0 (research project)"


def make_fetcher(timeout: int):
    try:
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        def fetch(url: str) -> bytes:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content

        return fetch
    except ImportError:
        import urllib.request

        print("[uwaga] brak 'requests' -> urllib (wolniej)")

        def fetch(url: str) -> bytes:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()

        return fetch


def candidates(url: str) -> List[str]:
    """Do adnotacji wystarczy wariant 'large' z iNaturalist (~1024 px) zamiast
    oryginalu wazacego kilka MB. Gdyby go nie bylo, leci oryginal."""
    if "inaturalist-open-data" in url:
        name = url.rsplit("/", 1)[-1]
        if name.startswith("original."):
            return [url.replace("/" + name, "/large." + name.split(".", 1)[1]), url]
    return [url]


def safe_name(rank: str, gbif_id: str, url: str) -> str:
    base = gbif_id.strip() or re.sub(r"\W+", "_", url.rsplit("/", 1)[-1])[:40]
    return f"{int(rank):04d}_{base}.jpg"


def download(rows: List[dict], out_dir: Path, workers: int, max_side: int, timeout: int, quality: int) -> Dict[str, str]:
    from PIL import Image
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    fetch = make_fetcher(timeout)
    failures: Dict[str, str] = {}

    todo = [r for r in rows if not (out_dir / r["_filename"]).exists()]
    print(f"[pobieranie] juz na dysku: {len(rows) - len(todo)}, do pobrania: {len(todo)}")

    def one(row: dict) -> None:
        dest = out_dir / row["_filename"]
        last = ""
        for cand in candidates(row["identifier"]):
            try:
                img = Image.open(io.BytesIO(fetch(cand))).convert("RGB")
                if max_side > 0:
                    img.thumbnail((max_side, max_side))
                img.save(dest, "JPEG", quality=quality)
                return
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:110]}"
        failures[row["identifier"]] = last

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(one, todo), total=len(todo), desc="Zdjecia"))

    return failures


def cleanup(out_dir: Path, process_all: bool) -> None:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    rejected_dir = out_dir / "rejected"

    if not images_dir.exists():
        raise SystemExit(f"Blad: nie ma {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".jpg")
    labeled = {p.stem for p in labels_dir.glob("*.txt") if p.stat().st_size > 0} if labels_dir.exists() else set()

    if not labeled:
        raise SystemExit(f"Blad: w {labels_dir} nie ma zadnych niepustych etykiet - nic nie ruszam.")

    # domyslnie ruszamy tylko to, co juz przejrzales: do ostatniego zdjecia z etykieta
    limit_rank = max(int(s.split("_", 1)[0]) for s in labeled if s.split("_", 1)[0].isdigit())
    rejected_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped_tail = 0
    for img in images:
        if img.stem in labeled:
            continue
        rank = int(img.stem.split("_", 1)[0]) if img.stem.split("_", 1)[0].isdigit() else 10**9
        if not process_all and rank > limit_rank:
            skipped_tail += 1
            continue
        target = rejected_dir / img.name
        if target.exists():
            target = rejected_dir / f"{img.stem}_dup{img.suffix}"
        shutil.move(str(img), str(target))
        empty_label = labels_dir / (img.stem + ".txt")
        if empty_label.exists():
            empty_label.unlink()
        moved += 1

    print(f"[cleanup] opisanych: {len(labeled)}")
    print(f"[cleanup] przeniesionych do {rejected_dir}: {moved}")
    if skipped_tail:
        print(f"[cleanup] nietkniete (jeszcze nieprzejrzane, ranking > {limit_rank}): {skipped_tail}")
        print("          zeby przerobic tez ogon: --cleanup --all")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="pobierz tylko N pierwszych wg rankingu (0 = wszystkie)")
    ap.add_argument("--max-side", type=int, default=1280, help="dluzszy bok zapisywanego zdjecia")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--class-name", default="ragwort")
    ap.add_argument("--cleanup", action="store_true", help="po adnotacji: nieopisane zdjecia do rejected/")
    ap.add_argument("--all", action="store_true", help="przy --cleanup rusz takze zdjecia jeszcze nieprzejrzane")
    args = ap.parse_args()

    if args.cleanup:
        cleanup(args.out, args.all)
        return

    if not args.csv.exists():
        raise SystemExit(f"Blad: nie ma {args.csv}")

    with open(args.csv, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    rank_col = "rank_score" if rows and "rank_score" in rows[0] else "rank"
    for i, r in enumerate(rows, start=1):
        r["_rank"] = r.get(rank_col) or str(i)
        r["_filename"] = safe_name(r["_rank"], r.get("gbifID", ""), r["identifier"])

    if args.limit > 0:
        rows = rows[: args.limit]

    images_dir = args.out / "images"
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    print(f"=== Przygotowanie do adnotacji: {len(rows)} zdjec z {args.csv.name} ===")

    failures = download(rows, images_dir, args.workers, args.max_side, args.timeout, args.quality)

    (args.out / "classes.txt").write_text(args.class_name + "\n", encoding="utf-8")

    with open(args.out / "index.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "rank", "gbifID", "identifier", "license", "rightsHolder", "pobrane"])
        for r in rows:
            w.writerow([r["_filename"], r["_rank"], r.get("gbifID", ""), r["identifier"],
                        r.get("license", ""), r.get("rightsHolder", ""),
                        int((images_dir / r["_filename"]).exists())])

    if failures:
        with open(args.out / "failed.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["identifier", "error"])
            w.writerows(failures.items())

    have = sum(1 for r in rows if (images_dir / r["_filename"]).exists())
    print(f"\n[gotowe] zdjec w {images_dir}: {have} (nieudanych: {len(failures)})")
    print(f"[gotowe] pusty folder na etykiety: {args.out / 'labels'}")
    print(f"[gotowe] klasy: {args.out / 'classes.txt'}  (zawartosc: {args.class_name})")
    print("\nW labelImg: Open Dir -> images, Change Save Dir -> labels, format YOLO, View -> Auto Save Mode.")
    print("Po adnotacji:  python src/scripts/prepare_labeling.py --cleanup")


if __name__ == "__main__":
    main()
