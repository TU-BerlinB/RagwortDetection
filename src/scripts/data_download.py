"""
data_download.py

Pobiera dwa zbiory danych do detekcji starca (ragwort / Jakobskreuzkraut)
z Roboflow Universe, łączy je w jeden zbiór ze wspólną listą klas i dzieli
wynik na podzbiory train/test.

Zbiory danych:
  - https://universe.roboflow.com/group-project-i4pjs/ragwort-detect
    (workspace: group-project-i4pjs, project: ragwort-detect, 50 zdjęć, 1 klasa: "ragwort")
  - https://universe.roboflow.com/jakobskreuzkraut/jakobskreuzkraut-lsgca
    (workspace: jakobskreuzkraut, project: jakobskreuzkraut-lsgca, 895 zdjęć, klasy: "Jakobskreuzkraut", "objects")

Wymagania
---------
pip install roboflow pyyaml

Potrzebny jest też darmowy klucz API Roboflow:
  1. Załóż konto na https://app.roboflow.com
  2. Wejdź w Settings -> API Keys, skopiuj "Private API Key"
  3. Ustaw zmienną środowiskową ROBOFLOW_API_KEY albo podaj --api-key

Uwaga na licencje:
  - ragwort-detect jest na licencji CC BY 4.0 -> przy publikacji/użyciu
    wyników wymagane jest podanie autora/źródła.
  - jakobskreuzkraut-lsgca jest na licencji CC0 (domena publiczna).

Użycie
------
python src/scripts/data_download.py --api-key TWOJ_KLUCZ
python src/scripts/data_download.py --test-size 0.15 --seed 123
python src/scripts/data_download.py --exclude-classes objects
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Brakuje zależności 'pyyaml'. Zainstaluj: pip install pyyaml"
    ) from exc

try:
    from roboflow import Roboflow
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Brakuje zależności 'roboflow'. Zainstaluj: pip install roboflow"
    ) from exc


# src/scripts/data_download.py -> src -> katalog główny projektu
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "combined_dataset"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_downloads"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSpec:
    key: str  # krótki prefiks używany w nazwach plików wyjściowych
    workspace: str
    project: str
    version: Optional[int] = None  # None -> użyj najnowszej dostępnej wersji


DATASETS: List[DatasetSpec] = [
    DatasetSpec(key="ragwort", workspace="group-project-i4pjs", project="ragwort-detect"),
    DatasetSpec(key="jkk", workspace="jakobskreuzkraut", project="jakobskreuzkraut-lsgca"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Klucz API Roboflow (albo ustaw zmienną środowiskową ROBOFLOW_API_KEY)",
    )
    parser.add_argument(
        "--format",
        default="yolov8",
        help="Format eksportu obsługiwany przez Roboflow, np. yolov8, yolov5pytorch, coco (domyślnie: yolov8)",
    )
    parser.add_argument(
        "--ragwort-version",
        type=int,
        default=None,
        help="Numer wersji zbioru ragwort-detect (domyślnie: najnowsza)",
    )
    parser.add_argument(
        "--jkk-version",
        type=int,
        default=None,
        help="Numer wersji zbioru jakobskreuzkraut-lsgca (domyślnie: najnowsza)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Udział danych trafiających do zbioru testowego (domyślnie: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości dla powtarzalnego podziału (domyślnie: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Gdzie zapisać połączony zbiór train/test",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Gdzie zapisać surowe pobrane dane (przed połączeniem)",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=[],
        help="Nazwy klas do całkowitego pominięcia (bez rozróżniania wielkości liter), np. --exclude-classes objects",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Zachowaj surowe, pobrane dane po połączeniu (domyślnie: są usuwane)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Wymagany jest klucz API Roboflow. Podaj --api-key albo ustaw zmienną ROBOFLOW_API_KEY."
        )
    if not 0 < args.test_size < 1:
        parser.error("--test-size musi być w przedziale (0, 1)")
    return args


def resolve_version(project, requested: Optional[int]):
    """Zwraca obiekt Version -- podaną wersję albo najnowszą dostępną."""
    if requested is not None:
        return project.version(requested)

    versions = project.versions()
    if not versions:
        raise RuntimeError(f"Nie znaleziono żadnej wersji dla projektu '{project.id}'")

    def version_number(v) -> int:
        # v.version bywa albo samą liczbą, albo pełnym identyfikatorem
        # w stylu "workspace/project/N" -- w obu przypadkach interesuje nas N.
        return int(str(v.version).split("/")[-1])

    latest = max(versions, key=version_number)
    return project.version(version_number(latest))


def download_dataset(rf: Roboflow, spec: DatasetSpec, fmt: str, raw_dir: Path) -> Path:
    print(f"[pobieranie] {spec.workspace}/{spec.project} (format={fmt}) ...")
    project = rf.workspace(spec.workspace).project(spec.project)
    version = resolve_version(project, spec.version)

    target_dir = raw_dir / spec.key
    if target_dir.exists():
        shutil.rmtree(target_dir)

    dataset = version.download(fmt, location=str(target_dir))
    print(f"[pobieranie] zapisano w {dataset.location}")
    return Path(dataset.location)


def load_class_names(dataset_dir: Path) -> List[str]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Nie znaleziono data.yaml w {dataset_dir}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    if not isinstance(names, list):
        raise ValueError(f"Nieoczekiwany format pola 'names' w {yaml_path}: {names!r}")
    return [str(n) for n in names]


def collect_samples(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Zbiera wszystkie pary obraz/etykieta ze splitów train/valid/test utworzonych przez Roboflow."""
    samples: List[Tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = labels_dir / (image_path.stem + ".txt")
            samples.append((image_path, label_path))
    return samples


CLASS_ALIASES: Dict[str, str] = {
    "jakobskreuzkraut": "ragwort",
    "starzec": "ragwort",
}


def build_global_classes(
    per_dataset_names: Dict[str, List[str]],
    exclude: List[str],
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, Dict[int, Optional[int]]]]:
    """
    Łączy listy klas wszystkich zbiorów w jedną, globalną listę (bez duplikatów,
    porównywanych bez rozróżniania wielkości liter, z uwzględnieniem synonimów)
    i buduje mapowanie lokalny_indeks -> globalny_indeks dla każdego zbioru.
    Wartość None oznacza "pomiń tę klasę" (była na liście --exclude-classes).
    """
    alias_map = aliases if aliases is not None else CLASS_ALIASES
    exclude_lower = {c.lower() for c in exclude}
    global_names: List[str] = []
    global_lookup: Dict[str, int] = {}
    mapping: Dict[str, Dict[int, Optional[int]]] = {}

    for key, names in per_dataset_names.items():
        mapping[key] = {}
        for local_idx, name in enumerate(names):
            canonical = alias_map.get(name.lower(), name)
            if canonical.lower() in exclude_lower or name.lower() in exclude_lower:
                mapping[key][local_idx] = None
                continue
            lookup_key = canonical.lower()
            if lookup_key not in global_lookup:
                global_lookup[lookup_key] = len(global_names)
                global_names.append(canonical)
            mapping[key][local_idx] = global_lookup[lookup_key]

    return global_names, mapping


def remap_label_file(src_label: Path, class_map: Dict[int, Optional[int]]) -> List[str]:
    """Wczytuje plik etykiet YOLO i zwraca linie z przemapowanymi indeksami klas."""
    if not src_label.exists():
        return []

    out_lines: List[str] = []
    with open(src_label, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            local_cls = int(parts[0])
            global_cls = class_map.get(local_cls)
            if global_cls is None:
                continue  # klasa pominięta przez --exclude-classes
            parts[0] = str(global_cls)
            out_lines.append(" ".join(parts))
    return out_lines


def write_split(
    samples: List[Tuple[str, Path, Path]],
    class_maps: Dict[str, Dict[int, Optional[int]]],
    split_name: str,
    output_dir: Path,
) -> int:
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for dataset_key, image_path, label_path in samples:
        out_name = f"{dataset_key}_{image_path.name}"
        shutil.copy2(image_path, images_out / out_name)

        remapped = remap_label_file(label_path, class_maps[dataset_key])
        label_out_path = labels_out / f"{dataset_key}_{image_path.stem}.txt"
        label_out_path.write_text(
            "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8"
        )
        written += 1
    return written


def write_data_yaml(output_dir: Path, class_names: List[str]) -> None:
    content = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    DATASETS[0].version = args.ragwort_version
    DATASETS[1].version = args.jkk_version

    rf = Roboflow(api_key=args.api_key)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pobranie obu zbiorów z Roboflow Universe
    dataset_dirs: Dict[str, Path] = {}
    for spec in DATASETS:
        dataset_dirs[spec.key] = download_dataset(rf, spec, args.format, args.raw_dir)

    # 2. Wspólna lista klas + mapowanie indeksów
    per_dataset_names = {key: load_class_names(d) for key, d in dataset_dirs.items()}
    global_classes, class_maps = build_global_classes(per_dataset_names, args.exclude_classes)
    print(f"[klasy] połączono {len(global_classes)} klas: {global_classes}")

    # 3. Zebranie wszystkich par obraz/etykieta z obu zbiorów (ignorując oryginalny split Roboflow)
    all_samples: List[Tuple[str, Path, Path]] = []
    for key, d in dataset_dirs.items():
        pairs = collect_samples(d)
        print(f"[dane] {key}: {len(pairs)} zdjęć")
        all_samples.extend((key, img, lbl) for img, lbl in pairs)

    if not all_samples:
        raise RuntimeError("Nie zebrano żadnych zdjęć z żadnego zbioru -- przerywam.")

    # 4. Losowy, powtarzalny podział na train/test
    random.shuffle(all_samples)
    n_test = max(1, round(len(all_samples) * args.test_size))
    test_samples = all_samples[:n_test]
    train_samples = all_samples[n_test:]

    n_train_written = write_split(train_samples, class_maps, "train", args.output_dir)
    n_test_written = write_split(test_samples, class_maps, "test", args.output_dir)
    write_data_yaml(args.output_dir, global_classes)

    print(f"[gotowe] train: {n_train_written} zdjęć, test: {n_test_written} zdjęć")
    print(f"[gotowe] zbiór zapisany w: {args.output_dir}")
    print(f"[gotowe] plik konfiguracyjny: {args.output_dir / 'data.yaml'}")

    if not args.keep_raw:
        print(f"[porządki] usuwam surowe pobrane dane z {args.raw_dir}")
        shutil.rmtree(args.raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()