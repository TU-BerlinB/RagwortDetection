"""
Wybor reprezentatywnych zdjec metoda facility location (greedy k-center) na embeddingach DINO.

KOMENDY - caly ten blok mozna wkleic do konsoli w /workspace, linie z # sa ignorowane:

# 1. sam skan danych, bez modelu (szybki test, ze sciezki sie zgadzaja):
python src/scripts/select_representative.py --dry-run

# 2. szybki test na 40 zdjeciach (sprawdza, czy model sie w ogole laduje):
python src/scripts/select_representative.py --limit 40 --n-select 10

# 3. wlasciwy przebieg na data/combined_dataset/train:
python src/scripts/select_representative.py --n-select 150

# 4. train + test + 1000 negatywow z data/plants, po rowno miedzy klasy:
python src/scripts/select_representative.py --split all --extra-dir data/plants --n-select 200 --equal-classes

# 5. to samo + kopie wybranych zdjec do outputs/selected_images/:
python src/scripts/select_representative.py --n-select 150 --copy-images

# 6. wymuszenie przeliczenia embeddingow (domyslnie brane z cache'u):
python src/scripts/select_representative.py --n-select 150 --recompute

MODEL: domyslnie skrypt probuje DINOv3, a gdy repo jest zamkniete (blad 401 / gated repo),
automatycznie przechodzi na publiczne facebook/dinov2-small. Zeby wymusic konkretny model:
python src/scripts/select_representative.py --model facebook/dinov2-base --n-select 150

# DOSTEP DO DINOv3 (opcjonalny) - najpierw kliknij "Agree and access repository" na
# https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m , potem w kontenerze:
pip install -U huggingface_hub
hf auth login          # starsze wersje: huggingface-cli login
# albo bez logowania:  export HF_TOKEN=hf_twoj_token

WEJSCIE:  data/combined_dataset/<split>/images + /labels   (nazwy klas z data.yaml)
WYJSCIE:  outputs/embeddings_combined_<split>.npz          cache embeddingow
          outputs/selected_representative.csv              lista zdjec do adnotacji
          outputs/selection_summary.json                   parametry + statystyki
          outputs/selected_images/<klasa>/                 tylko z --copy-images
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

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

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "combined_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RAGWORT_SYNONYMS = {"ragwort", "jakobskreuzkraut", "starzec", "senecio"}

CLASS_RAGWORT = "ragwort"
CLASS_OTHERS = "others"

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


@dataclass
class Sample:
    path: Path
    label: str

    @property
    def rel_path(self) -> str:
        try:
            return self.path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


def load_ragwort_class_ids(data_dir: Path) -> Set[int]:
    yaml_path = data_dir / "data.yaml"
    names: List[str] = []

    if yaml_path.exists():
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw = data.get("names", [])
            if isinstance(raw, dict):
                raw = [raw[k] for k in sorted(raw, key=int)]
            names = [str(n) for n in raw]
        except ImportError:
            print("[uwaga] brak pyyaml -> zakladam, ze klasa 0 to starzec")
        except Exception as exc:
            print(f"[uwaga] nie udalo sie wczytac {yaml_path}: {exc}")

    ids = {i for i, n in enumerate(names) if n.strip().lower() in RAGWORT_SYNONYMS}
    return ids or {0}


def label_of_image(label_path: Path, ragwort_ids: Set[int]) -> str:
    if not label_path.exists():
        return CLASS_OTHERS

    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                if int(float(parts[0])) in ragwort_ids:
                    return CLASS_RAGWORT
            except ValueError:
                continue
    return CLASS_OTHERS


def collect_samples(data_dir: Path, splits: Sequence[str]) -> List[Sample]:
    ragwort_ids = load_ragwort_class_ids(data_dir)
    samples: List[Sample] = []

    # Sprawdzenie czy podany katalog bezposrednio zawiera zdjecia (np. data/other_images)
    if data_dir.exists() and data_dir.is_dir():
        direct_paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if direct_paths:
            labels_dir = data_dir / "labels"
            print(f"[skan] bezposredni katalog {data_dir}: {len(direct_paths)} zdjec")
            for img in direct_paths:
                lbl_file = labels_dir / (img.stem + ".txt") if labels_dir.exists() else None
                lbl = label_of_image(lbl_file, ragwort_ids) if lbl_file and lbl_file.exists() else CLASS_OTHERS
                samples.append(Sample(img, lbl))
            return samples

    for split in splits:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"

        if not images_dir.exists():
            print(f"[uwaga] pomijam '{split}' - brak {images_dir}")
            continue

        paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for img in paths:
            samples.append(Sample(img, label_of_image(labels_dir / (img.stem + ".txt"), ragwort_ids)))
        print(f"[skan] {split}: {len(paths)} zdjec (ID klas starca: {sorted(ragwort_ids)})")

    return samples


def collect_extra_dir(extra_dir: Path, label: str) -> List[Sample]:
    if not extra_dir.exists():
        print(f"[uwaga] pomijam --extra-dir - brak {extra_dir}")
        return []

    paths = sorted(p for p in extra_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"[skan] {extra_dir}: {len(paths)} zdjec jako '{label}'")
    return [Sample(p, label) for p in paths]


def subsample(samples: List[Sample], limit: int) -> List[Sample]:
    if limit <= 0 or limit >= len(samples):
        return samples

    step = len(samples) / limit
    picked = [samples[min(len(samples) - 1, int(i * step))] for i in range(limit)]
    print(f"[limit] biore {len(picked)} z {len(samples)} zdjec (co ~{step:.1f}-te)")
    return picked


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


def compute_embeddings(samples: List[Sample], batch_size: int, model_name: str | None) -> Tuple[np.ndarray, str]:
    import torch
    from PIL import Image
    from tqdm import tqdm

    model, used_name = load_backbone(model_name)
    print(f"[model] uzywam {used_name} na: {model.device}")

    vectors: List[np.ndarray] = []
    for start in tqdm(range(0, len(samples), batch_size), desc="Embeddingi"):
        batch = samples[start : start + batch_size]
        images = [Image.open(s.path).convert("RGB") for s in batch]

        with torch.no_grad():
            features = model.extract_features(images)

        vectors.append(features.mean(dim=1).cpu().numpy())
        for img in images:
            img.close()

    return np.vstack(vectors).astype(np.float32), used_name


def load_or_compute_embeddings(
    samples: List[Sample], cache_path: Path, batch_size: int, recompute: bool, model_name: str | None
) -> Tuple[np.ndarray, str]:
    current = np.array([s.rel_path for s in samples])

    if cache_path.exists() and not recompute:
        cached = np.load(cache_path, allow_pickle=False)
        cached_model = str(cached["model"]) if "model" in cached else "?"
        same_files = "paths" in cached and len(cached["paths"]) == len(current) and np.array_equal(cached["paths"], current)
        same_model = model_name is None or cached_model == model_name

        if same_files and same_model:
            print(f"[cache] wczytuje embeddingi z {cache_path} (model: {cached_model})")
            return cached["embeddings"].astype(np.float32), cached_model
        print("[cache] zmienila sie lista plikow albo model -> liczenie od nowa")

    embeddings, used_name = compute_embeddings(samples, batch_size, model_name)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        paths=current,
        labels=np.array([s.label for s in samples]),
        model=np.array(used_name),
    )
    print(f"[cache] zapisano {cache_path} (ksztalt: {embeddings.shape})")

    return embeddings, used_name


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


def allocate_per_class(counts: Dict[str, int], n_select: int, equal: bool) -> Dict[str, int]:
    classes = sorted(counts)
    total = sum(counts.values())
    if total == 0:
        return {c: 0 for c in classes}

    if equal:
        base = n_select // len(classes)
        alloc = {c: min(base, counts[c]) for c in classes}
    else:
        alloc = {c: min(counts[c], int(round(n_select * counts[c] / total))) for c in classes}

    while sum(alloc.values()) < n_select and any(alloc[c] < counts[c] for c in classes):
        alloc[max(classes, key=lambda k: counts[k] - alloc[k])] += 1
    while sum(alloc.values()) > n_select:
        alloc[max(classes, key=lambda k: alloc[k])] -= 1

    return alloc


def write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "filename", "label", "pick_order", "gap_at_pick", "radius_after"])
        writer.writeheader()
        writer.writerows(rows)


def copy_selected(rows: List[dict], target_dir: Path, preserve_names: bool = False) -> None:
    for row in rows:
        dest_dir = target_dir / row["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = row["filename"] if preserve_names else f"{row['pick_order']:03d}_{row['filename']}"
        shutil.copy2(REPO_ROOT / row["path"], dest_dir / dest_name)
    print(f"[kopie] zdjecia w {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--extra-dir", type=Path, default=None, help="dodatkowy plaski folder zdjec, np. data/plants")
    parser.add_argument("--extra-label", default=CLASS_OTHERS)
    parser.add_argument("--n-select", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="uzyj tylko N zdjec z puli (0 = wszystkie)")
    parser.add_argument("--model", default=None, help="np. facebook/dinov2-small (domyslnie: DINOv3, a gdy gated -> DINOv2)")
    parser.add_argument("--equal-classes", action="store_true")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--start", default="medoid", choices=["medoid", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--copy-dir", type=Path, default=None, help="opcjonalny folder docelowy dla kopii wybranych zdjec")
    parser.add_argument("--preserve-filenames", action="store_true", help="nie dodawaj prefiksu 001_ do nazw plikow przy kopiowaniu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]

    print("=== Wybor reprezentatywnych zdjec (facility location / k-center) ===")
    print(f"Dane: {args.data_dir} (split: {', '.join(splits)})")

    samples = collect_samples(args.data_dir, splits)
    if args.extra_dir is not None:
        samples += collect_extra_dir(args.extra_dir, args.extra_label)

    if not samples:
        raise SystemExit(f"Blad: brak zdjec w {args.data_dir}. Sprawdz, czy istnieje {args.data_dir}/{splits[0]}/images.")

    samples = subsample(samples, args.limit)

    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1

    print(f"[pula] {len(samples)} zdjec: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for label, n in sorted(counts.items()):
        if n < 50 and args.extra_dir is None and args.limit <= 0:
            print(f"[uwaga] klasa '{label}' ma tylko {n} zdjec - rozwaz: --extra-dir data/plants --equal-classes")

    alloc = allocate_per_class(counts, args.n_select, args.equal_classes)
    print("[podzial] do wyboru: " + ", ".join(f"{k}={v}" for k, v in sorted(alloc.items())))

    if args.dry_run:
        print("\n[dry-run] koniec - bez embeddingow i selekcji.")
        return

    suffix = f"_limit{args.limit}" if args.limit > 0 else ""
    dataset_tag = args.data_dir.name.replace(" ", "_")
    cache_name = f"embeddings_{dataset_tag}_{'_'.join(splits)}{suffix}.npz"
    embeddings, used_model = load_or_compute_embeddings(
        samples, args.output_dir / cache_name, args.batch_size, args.recompute, args.model
    )
    X = prepare_metric_space(embeddings, args.metric)

    rows: List[dict] = []
    per_class: Dict[str, dict] = {}

    for label in sorted(counts):
        pool_idx = [i for i, s in enumerate(samples) if s.label == label]
        if alloc[label] <= 0:
            continue

        selected, gaps, radii = greedy_k_center(X[pool_idx], alloc[label], args.start, args.seed)

        for order, (local, gap, radius) in enumerate(zip(selected, gaps, radii), start=1):
            sample = samples[pool_idx[local]]
            rows.append(
                {
                    "path": sample.rel_path,
                    "filename": sample.path.name,
                    "label": label,
                    "pick_order": order,
                    "gap_at_pick": "" if gap == float("inf") else round(gap, 6),
                    "radius_after": round(radius, 6),
                }
            )

        per_class[label] = {
            "pool": counts[label],
            "selected": len(selected),
            "start_radius": round(radii[0], 6),
            "final_radius": round(radii[-1], 6),
        }
        print(f"[k-center] {label}: {len(selected)}/{counts[label]}, promien {radii[0]:.4f} -> {radii[-1]:.4f}")

    csv_path = args.output_dir / "selected_representative.csv"
    write_csv(rows, csv_path)

    summary_path = args.output_dir / "selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "splits": splits,
                "extra_dir": str(args.extra_dir) if args.extra_dir else None,
                "model": used_model,
                "metric": args.metric,
                "start": args.start,
                "limit": args.limit,
                "n_select_requested": args.n_select,
                "n_select_actual": len(rows),
                "pool_counts": counts,
                "allocation": alloc,
                "per_class": per_class,
                "embeddings_cache": str(args.output_dir / cache_name),
                "csv": str(csv_path),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    if args.copy_images or args.copy_dir is not None:
        target_dir = args.copy_dir if args.copy_dir is not None else (args.output_dir / "selected_images")
        copy_selected(rows, target_dir, preserve_names=args.preserve_filenames)

    print(f"\nGotowe. Lista do adnotacji: {csv_path}")
    print(f"Podsumowanie: {summary_path}")


if __name__ == "__main__":
    main()
