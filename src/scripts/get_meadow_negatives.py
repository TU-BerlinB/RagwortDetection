"""
Pobiera zbior z Zenodo (drony, 11 m nad laka pod Hamburgiem, 3,18 mm/piksel) i rozklada go na:
  - negatywy YOLO (zdjecia samej laki, pusty plik .txt = brak obiektu)
  - wyciete rosliny starca, uzywane pozniej jako pierwszy plan w generatorze syntetykow

Zbior: https://zenodo.org/records/12207476  (CC BY 4.0)
895 zdjec Jacobaea vulgaris + 9141 zdjec laki. Archiwum ma 810 MB.

KOMENDY - uruchamiaj tam, gdzie jest internet, z /workspace:

# 1. pobranie i rozpakowanie (raz, ~810 MB):
python src/scripts/get_meadow_negatives.py

# 2. gdy archiwum juz masz na dysku:
python src/scripts/get_meadow_negatives.py --zip data/DATENSATZ_DLS_01.zip

# 3. tylko podglad zawartosci archiwum, bez rozpakowywania:
python src/scripts/get_meadow_negatives.py --inspect

WYJSCIE:  data/meadow_negatives/images/  + labels/   negatywy gotowe do YOLO (puste .txt)
          data/ragwort_cutouts/                      wyciete rosliny (pierwszy plan)
          data/meadow_negatives/ATTRIBUTION.txt      wymagana atrybucja CC BY 4.0
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ZIP_URL = "https://zenodo.org/records/12207476/files/DATENSATZ_DLS_01.zip?download=1"
DEFAULT_ZIP = REPO_ROOT / "data" / "DATENSATZ_DLS_01.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ATTRIBUTION = """Zrodlo negatywow i wycietych roslin:
Jacobaea vulgaris and meadow image classification dataset (binary)
https://zenodo.org/records/12207476
Licencja: CC BY 4.0 - przy publikacji wynikow wymagane podanie zrodla.
Zdjecia z oktokoptera, wrzesien 2018, laki miejskie w Hamburgu, ok. 11 m wysokosci,
rozdzielczosc terenowa ok. 3,18 mm/piksel.

Uwaga autorow: zbior zawiera obrazy augmentowane (obroty, kadrowania, filtry),
wiec nie nadaje sie na zbior TESTOWY - grozi przeciekiem danych.
"""


def download(url: str, dest: Path, timeout: int) -> None:
    if dest.exists():
        print(f"[pobieranie] archiwum juz jest: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[pobieranie] {url}")

    try:
        import requests

        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
    except ImportError:
        import urllib.request

        urllib.request.urlretrieve(url, tmp)

    tmp.rename(dest)
    print(f"[pobieranie] zapisano {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def group_entries(zf: zipfile.ZipFile) -> dict:
    groups = collections.defaultdict(list)
    for info in zf.infolist():
        if info.is_dir():
            continue
        p = Path(info.filename)
        if p.suffix.lower() in IMAGE_SUFFIXES:
            groups[str(p.parent)].append(info.filename)
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--url", default=ZIP_URL)
    ap.add_argument("--out-neg", type=Path, default=REPO_ROOT / "data" / "meadow_negatives")
    ap.add_argument("--out-pos", type=Path, default=REPO_ROOT / "data" / "ragwort_cutouts")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--inspect", action="store_true", help="tylko pokaz, co jest w archiwum")
    ap.add_argument("--keep-zip", action="store_true", help="nie usuwaj archiwum po rozpakowaniu")
    args = ap.parse_args()

    download(args.url, args.zip, args.timeout)

    with zipfile.ZipFile(args.zip) as zf:
        groups = group_entries(zf)
        if not groups:
            raise SystemExit("Blad: w archiwum nie ma zdjec.")

        print("\n[archiwum] katalogi ze zdjeciami:")
        for folder, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"   {len(names):6d}  {folder or '(korzen)'}")

        if args.inspect:
            return

        # rozpoznanie po licznosci: 9141 laka, 895 starzec
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        meadow_folder, meadow_files = ordered[0]
        pos_folder, pos_files = (ordered[1] if len(ordered) > 1 else ("", []))
        print(f"\n[rozpoznanie] laka (negatywy): '{meadow_folder}' -> {len(meadow_files)} zdjec")
        print(f"[rozpoznanie] starzec:          '{pos_folder}' -> {len(pos_files)} zdjec")

        neg_images = args.out_neg / "images"
        neg_labels = args.out_neg / "labels"
        for d in (neg_images, neg_labels, args.out_pos):
            d.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(meadow_files, start=1):
            target = neg_images / f"meadow_{i:05d}{Path(name).suffix.lower()}"
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            (neg_labels / (target.stem + ".txt")).write_text("", encoding="utf-8")
            if i % 2000 == 0:
                print(f"   negatywy: {i}/{len(meadow_files)}", flush=True)

        for i, name in enumerate(pos_files, start=1):
            target = args.out_pos / f"ragwort_{i:05d}{Path(name).suffix.lower()}"
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    (args.out_neg / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    (args.out_pos / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    if not args.keep_zip:
        args.zip.unlink()
        print(f"[porzadki] usunieto {args.zip.name} (zostaw go flaga --keep-zip)")

    print(f"\n[gotowe] negatywy: {args.out_neg} (images + puste labels)")
    print(f"[gotowe] wyciete rosliny: {args.out_pos}")
    print("[gotowe] atrybucja CC BY 4.0 zapisana w ATTRIBUTION.txt")
    print("\nNastepny krok: python src/scripts/make_synthetic.py")


if __name__ == "__main__":
    main()
