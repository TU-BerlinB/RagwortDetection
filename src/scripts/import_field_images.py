"""
plantsextra.py

Uzupełnia zbiór danych o zdjęcia łąki / trawy robione z góry, BEZ starca
(obrazy tła / negatywy). Każde zapisane zdjęcie dostaje pusty plik .txt
o tej samej nazwie (format YOLO: brak ramek = brak obiektów).

Źródło zdjęć: LUCAS Cover photos (Komisja Europejska, JRC), licencja CC BY 4.0
  d'Andrimont R. i in. (2022), LUCAS cover photos 2006-2018 over the EU,
  Earth Syst. Sci. Data 14, 4463-4478, https://doi.org/10.5194/essd-14-4463-2022
  -> przy publikacji wymagane jest podanie źródła; pochodzenie każdego pliku
     zapisywane jest w metadata/plants_sources.csv

Struktura w katalogu głównym projektu:
    data/
      +-- plants/            -> zdjęcia .jpg + puste .txt (folder data/ tworzony, jeśli go nie ma)
    metadata/                -> tabela LUCAS, cache, logi, źródła (poza data/)

Jak wybierane są zdjęcia
------------------------
  - tylko punkty LUCAS sklasyfikowane jako tereny trawiaste (E10/E20/E30),
  - odrzucane są zdjęcia z niebem u góry kadru (robione pod kątem), z małą ilością
    roślinności, z czerwoną tablicą LUCAS oraz z wyraźnymi skupiskami żółtych
    kwiatów (żeby do negatywów nie trafił starzec),
  - jeśli zainstalowane są torch + transformers, model CLIP dodatkowo ocenia,
    czy zdjęcie jest zrobione z góry,
  - zapisywany jest środek kadru (domyślnie 60% szerokości i wysokości).

Wymagania
---------
pip install requests pillow numpy
(opcjonalnie, dokładniejszy filtr) pip install torch transformers

Użycie
------
python src/scripts/plantsextra.py
python src/scripts/plantsextra.py --target 1000 --countries PL,CZ,DE
python src/scripts/plantsextra.py --crop 0.5 --save-rejected
python src/scripts/plantsextra.py --sync-txt      # po ręcznym usunięciu zdjęć
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Brakuje zależności '{exc.name}'. Zainstaluj: pip install requests pillow numpy"
    ) from exc

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass


# src/scripts/plantsextra.py -> src -> katalog główny projektu
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

# Kraje o podobnych łąkach jak w Polsce (zmiana: --countries)
DEFAULT_COUNTRIES = "PL,CZ,SK,DE,LT,LV,EE,AT,HU"

# Progi filtra - dobrane na próbce 80 zdjęć LUCAS 2018 z Polski
SKY_TOP_PART = 0.30       # w jakiej górnej części kadru szukamy nieba
SKY_BLOCK = 16            # rozmiar bloku (px) na obrazku 320x240
SKY_STD = 12.0            # niebo jest gładkie: odchylenie jasności w bloku < SKY_STD
SKY_LUM = 110.0           # ... i jasne
MAX_SKY = 0.03            # maks. udział bloków "nieba" w górnej części kadru
VEG_SAT = 0.08            # piksel "roślinny": nasycenie >= VEG_SAT
VEG_HUE = (20.0, 170.0)   # ... i odcień od żółtego do zielono-niebieskiego (stopnie)
MIN_VEG_CLIP = 0.25       # min. udział roślinności w środku kadru (tryb clip)
MIN_VEG_SIMPLE = 0.50     # min. udział roślinności w środku kadru (tryb simple)
MAX_TOP_BOTTOM_DIFF = 60  # tryb simple: maks. różnica koloru góra/dół (zdjęcia pod kątem)
MAX_RED = 0.01            # czerwone tablice LUCAS leżące na trawie
MAX_YELLOW = 0.008        # skupiska jasnożółtych kwiatów (możliwy starzec)
REVIEW_YELLOW = 0.002     # powyżej tego progu zdjęcie warto obejrzeć ręcznie
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


# --------------------------------------------------------------------------- argumenty

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder na zdjęcia i puste .txt (domyślnie: data/plants)")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="Osobny folder na pliki .txt (domyślnie: ten sam co zdjęcia)")
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR,
                        help="Folder na metadane, cache i logi (domyślnie: metadata/ w katalogu projektu)")
    parser.add_argument("--target", type=int, default=1000,
                        help="Ile zdjęć ma być w folderze wynikowym (domyślnie: 1000)")
    parser.add_argument("--countries", default=DEFAULT_COUNTRIES,
                        help=f"Kody krajów po przecinku albo ALL (domyślnie: {DEFAULT_COUNTRIES})")
    parser.add_argument("--years", default="2018",
                        help="Lata badania LUCAS, np. 2018 albo 2015,2018 (domyślnie: 2018)")
    parser.add_argument("--classes", default="E10,E20,E30",
                        help="Klasy pokrycia terenu LUCAS; E = wszystkie trawiaste (domyślnie: E10,E20,E30)")
    parser.add_argument("--crop", type=float, default=0.6,
                        help="Jaka część środka kadru zostaje, 0.2-1.0; 1.0 = bez przycinania (domyślnie: 0.6)")
    parser.add_argument("--max-side", type=int, default=0,
                        help="Zmniejsz dłuższy bok zapisywanego zdjęcia do tylu pikseli (0 = bez zmian)")
    parser.add_argument("--filter", choices=["auto", "clip", "simple", "none"], default="auto",
                        help="auto = CLIP, jeśli zainstalowane torch + transformers, inaczej simple")
    parser.add_argument("--clip-model", default=CLIP_MODEL,
                        help=f"Model CLIP z Hugging Face (domyślnie: {CLIP_MODEL})")
    parser.add_argument("--clip-threshold", type=float, default=CLIP_THRESHOLD,
                        help=f"Min. pewność CLIP, że zdjęcie jest z góry (domyślnie: {CLIP_THRESHOLD})")
    parser.add_argument("--max-yellow", type=float, default=MAX_YELLOW,
                        help=f"Maks. udział jasnożółtych kwiatów w kadrze; 1 = wyłącz ten filtr "
                             f"(domyślnie: {MAX_YELLOW})")
    parser.add_argument("--workers", type=int, default=8,
                        help="Liczba równoległych pobrań (domyślnie: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Ziarno losowości kolejności zdjęć (domyślnie: 42)")
    parser.add_argument("--save-rejected", action="store_true",
                        help="Zapisuj odrzucone zdjęcia do metadata/plants_odrzucone (do przejrzenia)")
    parser.add_argument("--sync-txt", action="store_true",
                        help="Tylko uzupełnij brakujące .txt / usuń osierocone i zakończ")
    parser.add_argument("--base-url", default=LUCAS_BASE_URL, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop musi być w przedziale 0.2-1.0")
    if args.target < 1:
        parser.error("--target musi być dodatni")
    if args.workers < 1:
        parser.error("--workers musi być dodatni")
    return args


# --------------------------------------------------------------------------- sieć

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
                            print(f"\r[pobieranie] {desc}: {done / 1e6:.0f} / {total / 1e6:.0f} MB",
                                  end="", flush=True)
            print()
            if total and done != total:
                raise IOError(f"pobrano {done} z {total} bajtów")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            print(f"\n[pobieranie] błąd ({attempt}/3): {exc}")
            time.sleep(3 * attempt)
    raise SystemExit(f"Nie udało się pobrać {url}")


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


# --------------------------------------------------------------------------- metadane LUCAS

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
        print(f"[metadane] kandydaci z cache: {len(cands)} ({cache_path.name})")
    else:
        print("[metadane] przeszukuję tabelę LUCAS (1,3 GB CSV w archiwum) - to potrwa 1-3 minuty...")
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
                    raise SystemExit(f"Nieoczekiwany format tabeli LUCAS, brak kolumn: {missing}")
                i_id, i_year, i_cc = ix["point_id"], ix["year"], ix["nuts0"]
                i_lc, i_url = ix["lc1"], ix["file_path_ftp_cover"]
                i_lab, i_date = ix.get("lc1_label"), ix.get("survey_date")
                for n, row in enumerate(reader, 1):
                    if n % 50000 == 0:
                        print(f"\r[metadane] wierszy: {n:,}  kandydatów: {len(cands):,}  "
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
        for cand in cands:  # ten sam punkt może wystąpić kilka razy
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
        print(f"[metadane] pasujących zdjęć: {len(cands)} ({time.time() - t0:.0f} s)")

    if base_url.rstrip("/") != LUCAS_BASE_URL:
        for cand in cands:
            cand["url"] = cand["url"].replace(LUCAS_BASE_URL, base_url.rstrip("/"))
    return cands


# --------------------------------------------------------------------------- analiza obrazu

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
    """Udział jasnożółtych plamek wyraźnie jaśniejszych od otoczenia (kwiaty)."""
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
    """Proste miary: niebo u góry kadru, roślinność w środku, różnica koloru
    góra/dół, czerwień (tablice) i żółte kwiaty."""
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
    """Zero-shot CLIP: suma prawdopodobieństw opisów 'zdjęcie z góry'."""

    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[clip] ładuję model '{model_name}' ({self.device}); "
              "za pierwszym razem pobiera ok. 600 MB...")
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
            raise SystemExit("Tryb --filter clip wymaga: pip install torch transformers")
        print("[filtr] UWAGA: brak torch/transformers - używam prostszego filtra (mniej dokładny).\n"
              "        Dla lepszych wyników: pip install torch transformers")
        return None


def decide(stats: Dict[str, float], clip_score: Optional[float], mode: str,
           clip_threshold: float, max_yellow: float) -> Tuple[bool, str]:
    if mode == "none":
        return True, ""
    if stats["red"] > MAX_RED:
        return False, "czerwona tablica"
    if stats["sky"] > MAX_SKY:
        return False, f"niebo {stats['sky']:.2f}"
    if stats["yellow"] > max_yellow:
        return False, f"żółte kwiaty {stats['yellow']:.4f}"
    if clip_score is not None:
        if stats["veg"] < MIN_VEG_CLIP:
            return False, f"mało roślinności {stats['veg']:.2f}"
        if clip_score < clip_threshold:
            return False, f"clip {clip_score:.2f}"
        return True, ""
    if stats["veg"] < MIN_VEG_SIMPLE:
        return False, f"mało roślinności {stats['veg']:.2f}"
    if stats["diff"] > MAX_TOP_BOTTOM_DIFF:
        return False, f"zdjęcie pod kątem {stats['diff']:.0f}"
    return True, ""


# --------------------------------------------------------------------------- pliki wyjściowe

def sync_txt(img_dir: Path, lbl_dir: Path) -> Tuple[int, int, int]:
    """Dla zdjęć z tego skryptu (lucasRRRR_KK_ID.jpg) tworzy brakujące puste .txt
    i usuwa puste .txt, których zdjęcie skasowano. Innych plików nie rusza."""
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


# --------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()

    out_dir: Path = args.output_dir
    lbl_dir: Path = args.labels_dir or out_dir
    meta_dir: Path = args.meta_dir
    rej_dir = meta_dir / "plants_odrzucone"
    processed_path = meta_dir / "plants_przetworzone.csv"
    sources_path = meta_dir / "plants_sources.csv"
    review_path = meta_dir / "plants_do_przejrzenia.txt"

    out_dir.mkdir(parents=True, exist_ok=True)   # tworzy też data/, jeśli go nie ma
    meta_dir.mkdir(parents=True, exist_ok=True)

    have, created, removed = sync_txt(out_dir, lbl_dir)
    if created or removed:
        print(f"[txt] dodano {created}, usunięto {removed} osieroconych plików .txt")
    if args.sync_txt:
        print(f"[gotowe] zdjęć w {rel(out_dir)}: {have}, każde ma swój pusty plik .txt")
        return

    print(f"[start] zdjęcia: {rel(out_dir)} | metadane: {rel(meta_dir)}")
    print(f"[start] jest już {have} zdjęć, cel: {args.target}")
    if have >= args.target:
        print("[gotowe] cel osiągnięty - nic do zrobienia")
        return

    years = {y.strip() for y in args.years.split(",") if y.strip()}
    countries = (set() if args.countries.strip().upper() == "ALL"
                 else {c.strip().upper() for c in args.countries.split(",") if c.strip()})
    classes = [c.strip().upper() for c in args.classes.split(",") if c.strip()]

    zip_path = meta_dir / "lucas_cover_attr.csv.zip"
    if not zip_path.exists():
        print("[metadane] pobieram tabelę LUCAS (ok. 145 MB, tylko za pierwszym razem)")
        download_file(f"{args.base_url.rstrip('/')}/{LUCAS_META_ZIP}", zip_path, "tabela LUCAS")

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
    print(f"[dane] do sprawdzenia: {len(todo)} (sprawdzonych wcześniej: {len(processed)})")
    if not todo:
        print("[dane] brak nowych kandydatów - dodaj kraje (--countries) albo lata (--years 2015,2018)")
        return

    scorer = make_scorer(args.filter, args.clip_model)
    mode = args.filter if args.filter != "auto" else ("clip" if scorer else "simple")
    print(f"[filtr] tryb: {mode}")

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

                # 1. pobranie i wstępna analiza
                items: List[list] = []
                for cand, data, err in pool.map(fetch, batch):
                    if err:
                        errors += 1
                        proc_log.write({"key": cand_key(cand), "status": "błąd", "info": err})
                        continue
                    try:
                        img = Image.open(io.BytesIO(data))
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        img = trim_black_borders(img)
                        if min(img.size) < 200:
                            raise ValueError(f"za mały obraz {img.size}")
                    except Exception as exc:
                        errors += 1
                        proc_log.write({"key": cand_key(cand), "status": "błąd",
                                        "info": f"obraz: {str(exc)[:80]}"})
                        continue
                    items.append([cand, img, image_stats(img, args.crop), None])

                # 2. CLIP tylko dla zdjęć, których prosty test nie odrzucił
                if scorer:
                    need = [it for it in items
                            if decide(it[2], 1.0, "clip", 0.0, args.max_yellow)[0]]
                    for i in range(0, len(need), 16):
                        chunk = need[i: i + 16]
                        for it, score in zip(chunk, scorer.scores([it[1] for it in chunk])):
                            it[3] = score

                # 3. decyzja i zapis
                for cand, img, st, clip_score in items:
                    if accepted >= args.target:
                        break  # bez wpisu do logu - zostaną na następny raz
                    checked += 1
                    use_clip = clip_score if mode == "clip" else None
                    if mode == "clip" and clip_score is None:
                        use_clip = -1.0  # odrzucone już prostym testem
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
                                fh.write(f"{name}\tzolte={st['yellow']:.4f}\n")
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
                        proc_log.write({"key": cand_key(cand), "status": "odrzucone", "info": why})
                        if args.save_rejected:
                            small = img.copy()
                            small.thumbnail((800, 800))
                            small.save(rej_dir / name, "JPEG", quality=85)

                elapsed = (time.time() - t0) / 60
                print(f"\r[postęp] zapisane: {accepted}/{args.target} | sprawdzone: {checked} | "
                      f"odrzucone: {rejected} | błędy: {errors} | {elapsed:.1f} min   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stop] przerwano - uruchom skrypt ponownie, żeby kontynuować")
    finally:
        print()
        proc_log.close()
        sources.close()

    have, _, _ = sync_txt(out_dir, lbl_dir)
    print(f"[gotowe] zdjęć w {rel(out_dir)}: {have} (każde z pustym .txt)")
    print(f"[gotowe] źródła zdjęć (licencja CC BY 4.0): {rel(sources_path)}")
    if to_review:
        print(f"[uwaga] {to_review} nowych zdjęć ma drobne żółte kwiaty - obejrzyj je, czy to nie starzec "
              f"(lista: {rel(review_path)})")
    if have < args.target:
        print("[uwaga] zabrakło kandydatów - dodaj kraje, np. --countries PL,CZ,SK,DE,AT,HU,FR,IT,"
              " albo lata, np. --years 2015,2018 --classes E")
    print("[wskazówka] po ręcznym usunięciu nietrafionych zdjęć uruchom z --sync-txt")


if __name__ == "__main__":
    main()
