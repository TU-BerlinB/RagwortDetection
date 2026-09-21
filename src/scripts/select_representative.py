"""
Dwustopniowy wybor reprezentatywnych zdjec metoda facility location (greedy k-center)
na embeddingach DINO, ze zbioru GBIF sample_10000.csv (kolumna 'identifier' = URL zdjecia).

10000 -> 2000 -> 200. Wyniki jako dwa pliki CSV w folderze afterfacilitylocation/.

KOMENDY - caly blok mozna wkleic do konsoli w /workspace, linie z # sa ignorowane:

# 1. sam skan CSV, bez pobierania i bez modelu:
python src/scripts/select_representative.py --dry-run

# 2. szybki test na 40 zdjeciach (sprawdza pobieranie i model):
python src/scripts/select_representative.py --limit 40 --n-first 20 --n-second 5

# 3. wlasciwy przebieg 10000 -> 2000 -> 200:
python src/scripts/select_representative.py

# 4. to samo, ale wiecej watkow pobierania (szybciej, gdy lacze wyrabia):
python src/scripts/select_representative.py --workers 32

# 5. ponowne liczenie embeddingow od zera (ignoruje cache):
python src/scripts/select_representative.py --recompute

MODEL: domyslnie probuje DINOv3, a gdy repo jest zamkniete (401 / gated), przechodzi
na publiczne facebook/dinov2-small. Wymuszenie konkretnego modelu:
python src/scripts/select_representative.py --model facebook/dinov2-base

WEJSCIE:  sample_10000.csv  (kolumny GBIF: gbifID, identifier, license, rightsHolder, ...)
WYJSCIE:  afterfacilitylocation/selected_2000.csv    pierwszy etap
          afterfacilitylocation/selected_200.csv     drugi etap (podzbior tych 2000)
          afterfacilitylocation/embeddings.npz       cache embeddingow
          afterfacilitylocation/thumbs/              pobrane zdjecia (male kopie)
          afterfacilitylocation/failed.csv           URL-e, ktorych nie udalo sie pobrac

UWAGA: folder afterfacilitylocation/ potrafi urosnac do kilkuset MB - dopisz go do .gitignore.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_INPUT = REPO_ROOT / "sample_10000.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "afterfacilitylocation"

URL_COLUMN = "identifier"
ID_COLUMN = "gbifID"

MODEL_CANDIDATES = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov2-small",
]

HF_HINT = (
    "Model DINOv3 jest zamkniety (gated). Zeby go uzyc:\n"
    "  1) wejdz na https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
    "     i kliknij 'Agree and access repository'\n"
    "  2) w kontenerze:  pip install -U huggingface_hub  &&  hf auth login\n"
    "     (albo: export HF_TOKEN=hf_twoj_token)"
)

USER_AGENT = "RagwortDetection/1.0 (research project; contact via GitHub TU-BerlinB/RagwortDetection)"


def make_fetcher(timeout: int):
    """Funkcja pobierajaca bajty spod URL. Uzywa 'requests', a gdy go nie ma - urllib ze stdlib."""
    try:
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        def fetch(url: str) -> bytes:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content

        return fetch
    except ImportError:
        import urllib.request

        print("[uwaga] brak biblioteki 'requests' -> uzywam urllib (wolniej, bez wspoldzielenia polaczen).")
        print("        Szybciej bedzie po:  pip install requests")

        def fetch(url: str) -> bytes:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        return fetch


def download_candidates(url: str, prefer_small: bool) -> List[str]:
    """URL-e do sprobowania, od najlzejszego. iNaturalist trzyma warianty rozmiarowe
    pod ta sama sciezka, a 'original.jpg' potrafi wazyc kilka MB."""
    if prefer_small and "inaturalist-open-data" in url and url.endswith("/original.jpg"):
        return [url.replace("/original.jpg", "/medium.jpg"), url]
    return [url]


def read_rows(csv_path: Path, limit: int) -> Tuple[List[dict], List[str]]:
    if not csv_path.exists():
        raise SystemExit(f"Blad: nie ma pliku {csv_path}. Podaj inny przez --input.")

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        raw = list(reader)

    if URL_COLUMN not in fieldnames:
        raise SystemExit(f"Blad: w {csv_path.name} nie ma kolumny '{URL_COLUMN}'. Kolumny: {fieldnames}")

    rows: List[dict] = []
    seen = set()
    empty = 0
    dupes = 0

    for r in raw:
        url = (r.get(URL_COLUMN) or "").strip()
        if not url:
            empty += 1
            continue
        if url in seen:
            dupes += 1
            continue
        seen.add(url)
        rows.append(r)

    print(f"[csv] {csv_path.name}: {len(raw)} wierszy -> {len(rows)} unikalnych URL (puste: {empty}, duplikaty: {dupes})")

    if limit > 0 and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[min(len(rows) - 1, int(i * step))] for i in range(limit)]
        print(f"[limit] biore {len(rows)} zdjec")

    return rows, fieldnames


def thumb_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg")


def download_missing(
    rows: List[dict], cache_dir: Path, workers: int, max_side: int, timeout: int, prefer_small: bool
) -> Dict[str, str]:
    from PIL import Image
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not thumb_path(cache_dir, r[URL_COLUMN]).exists()]
    have = len(rows) - len(todo)
    print(f"[pobieranie] w cache: {have}, do pobrania: {len(todo)}")

    if not todo:
        return {}

    fetch_bytes = make_fetcher(timeout)
    failures: Dict[str, str] = {}

    def fetch(row: dict) -> None:
        url = row[URL_COLUMN]
        dest = thumb_path(cache_dir, url)
        last = ""
        for candidate in download_candidates(url, prefer_small):
            try:
                img = Image.open(io.BytesIO(fetch_bytes(candidate))).convert("RGB")
                img.thumbnail((max_side, max_side))
                img.save(dest, "JPEG", quality=90)
                return
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:120]}"
        failures[url] = last

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(fetch, todo), total=len(todo), desc="Pobieranie"))

    print(f"[pobieranie] nieudane: {len(failures)}")
    return failures


def load_cache(cache_path: Path, recompute: bool) -> Tuple[Dict[str, np.ndarray], str]:
    if recompute or not cache_path.exists():
        return {}, ""

    data = np.load(cache_path, allow_pickle=False)
    urls = data["urls"]
    vectors = data["embeddings"]
    model = str(data["model"]) if "model" in data else ""
    print(f"[cache] {len(urls)} embeddingow z {cache_path} (model: {model or '?'})")
    return {str(u): v for u, v in zip(urls, vectors)}, model


def save_cache(cache_path: Path, mapping: Dict[str, np.ndarray], model: str) -> None:
    if not mapping:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urls = list(mapping)
    np.savez(
        cache_path,
        urls=np.array(urls),
        embeddings=np.stack([mapping[u] for u in urls]).astype(np.float32),
        model=np.array(model),
    )


def load_backbone(model_name: str | None):
    from models.dinov3 import DINOv3

    if model_name:
        return DINOv3(model_name=model_name), model_name

    last_error = None
    for candidate in MODEL_CANDIDATES:
        try:
            print(f"[model] probuje: {candidate}")
            return DINOv3(model_name=candidate), candidate
        except Exception as exc:
            last_error = exc
            print(f"[model] {candidate} niedostepny ({type(exc).__name__}) -> probuje nastepny")
            if candidate.startswith("facebook/dinov3"):
                print(HF_HINT)

    raise SystemExit(f"Blad: nie udalo sie zaladowac zadnego modelu. Ostatni blad: {last_error}")


def embed_missing(
    rows: List[dict],
    cache_dir: Path,
    cache_path: Path,
    mapping: Dict[str, np.ndarray],
    model_name: str | None,
    batch_size: int,
    save_every: int,
) -> str:
    import torch
    from PIL import Image
    from tqdm import tqdm

    todo = [r for r in rows if r[URL_COLUMN] not in mapping and thumb_path(cache_dir, r[URL_COLUMN]).exists()]
    if not todo:
        print("[embeddingi] wszystko juz policzone")
        return ""

    model, used_name = load_backbone(model_name)
    print(f"[model] uzywam {used_name} na: {model.device}")

    since_save = 0
    for start in tqdm(range(0, len(todo), batch_size), desc="Embeddingi"):
        batch = todo[start : start + batch_size]
        images = []
        keep = []
        for row in batch:
            try:
                images.append(Image.open(thumb_path(cache_dir, row[URL_COLUMN])).convert("RGB"))
                keep.append(row)
            except Exception:
                continue

        if not images:
            continue

        with torch.no_grad():
            features = model.extract_features(images)

        pooled = features.mean(dim=1).cpu().numpy().astype(np.float32)
        for row, vec in zip(keep, pooled):
            mapping[row[URL_COLUMN]] = vec

        for img in images:
            img.close()

        since_save += len(keep)
        if save_every > 0 and since_save >= save_every:
            save_cache(cache_path, mapping, used_name)
            since_save = 0

    save_cache(cache_path, mapping, used_name)
    print(f"[cache] zapisano {cache_path} ({len(mapping)} wektorow)")
    return used_name


def prepare_metric_space(embeddings: np.ndarray, metric: str) -> np.ndarray:
    if metric != "cosine":
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def greedy_k_center(X: np.ndarray, p: int, start: str = "medoid", seed: int = 0) -> Tuple[List[int], List[float], List[float]]:
    n = X.shape[0]
    p = max(1, min(p, n))

    if start == "random":
        first = int(np.random.default_rng(seed).integers(n))
    else:
        first = int(np.argmin(np.linalg.norm(X - X.mean(axis=0), axis=1)))

    selected = [first]
    min_dist = np.linalg.norm(X - X[first], axis=1)
    gap_at_pick = [float("inf")]
    radius_after = [float(min_dist.max())]

    for _ in range(p - 1):
        nxt = int(np.argmax(min_dist))
        gap = float(min_dist[nxt])
        if gap <= 0.0:
            break

        selected.append(nxt)
        gap_at_pick.append(gap)
        min_dist = np.minimum(min_dist, np.linalg.norm(X - X[nxt], axis=1))
        radius_after.append(float(min_dist.max()))

    return selected, gap_at_pick, radius_after


def write_selection(
    rows: List[dict],
    fieldnames: List[str],
    order: List[int],
    gaps: List[float],
    radii: List[float],
    cache_dir: Path,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["pick_order", "gap_at_pick", "radius_after", "thumb_path"]
    columns = fieldnames + [c for c in extra if c not in fieldnames]

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for pick, (idx, gap, radius) in enumerate(zip(order, gaps, radii), start=1):
            row = dict(rows[idx])
            row["pick_order"] = pick
            row["gap_at_pick"] = "" if gap == float("inf") else round(gap, 6)
            row["radius_after"] = round(radius, 6)
            row["thumb_path"] = thumb_path(cache_dir, row[URL_COLUMN]).relative_to(REPO_ROOT).as_posix()
            writer.writerow(row)

    print(f"[zapis] {out_path}  ({len(order)} wierszy)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-first", type=int, default=2000, help="ile wybrac w pierwszym etapie")
    parser.add_argument("--n-second", type=int, default=200, help="ile wybrac z wyniku pierwszego etapu")
    parser.add_argument("--limit", type=int, default=0, help="uzyj tylko N wierszy z CSV (0 = wszystkie)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--start", default="medoid", choices=["medoid", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16, help="watki pobierania zdjec")
    parser.add_argument("--max-side", type=int, default=256, help="dluzszy bok zapisywanej miniatury")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-prefer-small", dest="prefer_small", action="store_false",
                        help="pobieraj zawsze oryginalny plik, nawet gdy jest wersja lzejsza")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=500, help="co ile zdjec zapisac cache embeddingow")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="tylko wczytanie CSV, bez pobierania i modelu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Facility location na zbiorze GBIF: 10000 -> 2000 -> 200 ===")
    rows, fieldnames = read_rows(args.input, args.limit)

    if not rows:
        raise SystemExit("Blad: brak wierszy z URL-em do zdjecia.")

    if args.dry_run:
        print(f"[dry-run] koniec. Do pobrania byloby {len(rows)} zdjec, etapy: {args.n_first} -> {args.n_second}")
        return

    cache_dir = args.output_dir / "thumbs"
    cache_path = args.output_dir / "embeddings.npz"

    failures = download_missing(rows, cache_dir, args.workers, args.max_side, args.timeout, args.prefer_small)
    if failures:
        fail_path = args.output_dir / "failed.csv"
        with open(fail_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["url", "error"])
            writer.writerows(failures.items())
        print(f"[zapis] {fail_path}  ({len(failures)} nieudanych)")

    mapping, cached_model = load_cache(cache_path, args.recompute)
    used_model = embed_missing(
        rows, cache_dir, cache_path, mapping, args.model, args.batch_size, args.save_every
    ) or cached_model

    usable = [r for r in rows if r[URL_COLUMN] in mapping]
    if not usable:
        raise SystemExit("Blad: nie udalo sie policzyc ani jednego embeddingu (sprawdz siec i model).")

    X = prepare_metric_space(np.stack([mapping[r[URL_COLUMN]] for r in usable]), args.metric)
    print(f"[pula] {len(usable)} zdjec z embeddingiem (model: {used_model or '?'}), wymiar: {X.shape[1]}")

    first_idx, first_gaps, first_radii = greedy_k_center(X, args.n_first, args.start, args.seed)
    print(f"[etap 1] {len(first_idx)} z {len(usable)}, promien {first_radii[0]:.4f} -> {first_radii[-1]:.4f}")

    X_first = X[first_idx]
    second_local, second_gaps, second_radii = greedy_k_center(X_first, args.n_second, args.start, args.seed)
    second_idx = [first_idx[i] for i in second_local]
    print(f"[etap 2] {len(second_idx)} z {len(first_idx)}, promien {second_radii[0]:.4f} -> {second_radii[-1]:.4f}")

    write_selection(usable, fieldnames, first_idx, first_gaps, first_radii, cache_dir,
                    args.output_dir / f"selected_{args.n_first}.csv")
    write_selection(usable, fieldnames, second_idx, second_gaps, second_radii, cache_dir,
                    args.output_dir / f"selected_{args.n_second}.csv")

    print(f"\nGotowe. Pliki w {args.output_dir}")


if __name__ == "__main__":
    main()
