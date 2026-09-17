"""
prepare_dino_data.py

Pobiera zbiory danych z Roboflow (lub korzysta z już pobranych) i dzieli zdjęcia
na dwa podfoldery do klasyfikacji / ekstrakcji embeddingów modelem DINOv3:
    data/dino_dataset/
        |-- ragwort/   -> zdjecia ZAWIERAJACE starca (ragwort / Jakobskreuzkraut)
        +-- others/    -> zdjecia BEZ starca (inne obiekty, rosliny lub samo tlo)

Struktura ta jest bezpośrednio kompatybilna z torchvision.datasets.ImageFolder
oraz modelem DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m).

Zbiory wejściowe:
  - https://universe.roboflow.com/group-project-i4pjs/ragwort-detect
  - https://universe.roboflow.com/jakobskreuzkraut/jakobskreuzkraut-lsgca

Użycie:
  python src/scripts/prepare_dino_data.py --api-key TWOJ_KLUCZ_ROBOFLOW
  python src/scripts/prepare_dino_data.py --from-combined data/combined_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dino_dataset"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_downloads"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Nazwy klas uznawane za starca (ragwort)
RAGWORT_SYNONYMS = {"ragwort", "jakobskreuzkraut", "starzec", "senecio"}


@dataclass
class DatasetSpec:
    key: str
    workspace: str
    project: str
    version: Optional[int] = None


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
        help="Klucz API Roboflow (lub ustaw zmienną środowiskową ROBOFLOW_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog wyjściowy z podfolderami 'ragwort' i 'others' (domyślnie: data/dino_dataset)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Katalog na surowe pobrania z Roboflow",
    )
    parser.add_argument(
        "--from-combined",
        type=Path,
        default=None,
        help="Opcjonalna ścieżka do już przygotowanego data/combined_dataset (omija ponowne pobieranie z Roboflow)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Zachowaj surowe pobrane dane po podziale (domyślnie: usuwane)",
    )
    return parser.parse_args()


def load_class_names(data_yaml_path: Path) -> List[str]:
    """Wczytuje listę klas z pliku data.yaml."""
    if not data_yaml_path.exists():
        return []
    with open(data_yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    return [str(n) for n in names]


def resolve_version(project, requested: Optional[int]):
    """Zwraca obiekt Version."""
    if requested is not None:
        return project.version(requested)
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"Brak wersji dla projektu '{project.id}'")

    def version_number(v) -> int:
        return int(str(v.version).split("/")[-1])

    latest = max(versions, key=version_number)
    return project.version(version_number(latest))


def download_raw_datasets(rf: Roboflow, raw_dir: Path) -> Dict[str, Path]:
    """Pobiera oba zbiory do katalogu raw_dir."""
    dataset_dirs = {}
    for spec in DATASETS:
        target_dir = raw_dir / spec.key
        if target_dir.exists():
            shutil.rmtree(target_dir)

        print(f"[pobieranie] {spec.workspace}/{spec.project}...")
        project = rf.workspace(spec.workspace).project(spec.project)
        version = resolve_version(project, spec.version)
        dataset = version.download("yolov8", location=str(target_dir))
        dataset_dirs[spec.key] = Path(dataset.location)
    return dataset_dirs


def has_ragwort_label(label_path: Path, ragwort_class_ids: Set[int]) -> bool:
    """Sprawdza, czy w pliku etykiet znajduje się przynajmniej jedna ramka ze starcem."""
    if not label_path.exists():
        return False

    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id in ragwort_class_ids:
                return True
    return False


def collect_images_and_labels(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Zbiera pary (ścieżka_do_obrazu, ścieżka_do_etykiety) ze wszystkich splitów w katalogu."""
    pairs = []
    # Sprawdzamy train/valid/test oraz bezpośrednio images/
    splits_to_check = [
        dataset_dir / "train",
        dataset_dir / "valid",
        dataset_dir / "test",
        dataset_dir,
    ]
    for sp in splits_to_check:
        images_dir = sp / "images" if (sp / "images").exists() else (sp if sp.name == "images" else None)
        if not images_dir or not images_dir.exists():
            continue
        labels_dir = images_dir.parent / "labels"

        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
                lbl = labels_dir / (img.stem + ".txt")
                pairs.append((img, lbl))
    return pairs


def split_into_classification(
    dataset_dirs: Dict[str, Path],
    output_dir: Path,
) -> Tuple[int, int]:
    """
    Kopiuje obrazy do katalogów:
      - output_dir / 'ragwort'  (jeśli zawierają starca)
      - output_dir / 'others'   (jeśli nie zawierają starca)
    """
    ragwort_out = output_dir / "ragwort"
    others_out = output_dir / "others"

    ragwort_out.mkdir(parents=True, exist_ok=True)
    others_out.mkdir(parents=True, exist_ok=True)

    ragwort_count = 0
    others_count = 0

    for dataset_key, d_dir in dataset_dirs.items():
        # Odczytujemy indeksy klas odpowiadających starcowi w tym zbiorze
        class_names = load_class_names(d_dir / "data.yaml")
        ragwort_ids: Set[int] = set()

        for idx, name in enumerate(class_names):
            if name.lower() in RAGWORT_SYNONYMS:
                ragwort_ids.add(idx)

        # Jeśli data.yaml nie zawierał nazw, a to zbiór ragwort-detect, domyślnie klasa 0 to ragwort
        if not ragwort_ids and dataset_key == "ragwort":
            ragwort_ids.add(0)

        pairs = collect_images_and_labels(d_dir)
        print(f"[przetwarzanie] zbiór '{dataset_key}': znaleziono {len(pairs)} zdjęć (ID klas starca: {ragwort_ids})")

        for img_path, lbl_path in pairs:
            has_rag = has_ragwort_label(lbl_path, ragwort_ids)
            out_filename = f"{dataset_key}_{img_path.name}"

            if has_rag:
                shutil.copy2(img_path, ragwort_out / out_filename)
                ragwort_count += 1
            else:
                shutil.copy2(img_path, others_out / out_filename)
                others_count += 1

    return ragwort_count, others_count


def split_from_combined_dataset(combined_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """Podział na bazie wcześniej połączonego data/combined_dataset."""
    ragwort_out = output_dir / "ragwort"
    others_out = output_dir / "others"

    ragwort_out.mkdir(parents=True, exist_ok=True)
    others_out.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(combined_dir / "data.yaml")
    ragwort_ids = {idx for idx, name in enumerate(class_names) if name.lower() in RAGWORT_SYNONYMS}
    if not ragwort_ids:
        ragwort_ids.add(0)  # domyślnie w combined_dataset klasa 0 to ragwort

    pairs = collect_images_and_labels(combined_dir)
    print(f"[przetwarzanie] zbiór combined ({combined_dir}): {len(pairs)} zdjęć (ID klas starca: {ragwort_ids})")

    ragwort_count = 0
    others_count = 0

    for img_path, lbl_path in pairs:
        has_rag = has_ragwort_label(lbl_path, ragwort_ids)
        out_filename = img_path.name

        if has_rag:
            shutil.copy2(img_path, ragwort_out / out_filename)
            ragwort_count += 1
        else:
            shutil.copy2(img_path, others_out / out_filename)
            others_count += 1

    return ragwort_count, others_count


def main():
    args = parse_args()
    print("=== Przygotowanie danych do klasyfikacji / DINOv3 ===")
    print(f"Docelowy katalog wyjściowy: {args.output_dir}")

    # Scenariusz 1: Użytkownik wskazał już istniejący combined_dataset
    if args.from_combined and args.from_combined.exists():
        print(f"[INFO] Korzystam z istniejącego zbioru: {args.from_combined}")
        n_rag, n_oth = split_from_combined_dataset(args.from_combined, args.output_dir)

    # Scenariusz 2: Sprawdzenie czy w data/combined_dataset już są pobrane dane
    elif (REPO_ROOT / "data" / "combined_dataset").exists() and any((REPO_ROOT / "data" / "combined_dataset").iterdir()):
        print(f"[INFO] Wykryto istniejące dane w data/combined_dataset. Używam ich bez ponownego pobierania.")
        n_rag, n_oth = split_from_combined_dataset(REPO_ROOT / "data" / "combined_dataset", args.output_dir)

    # Scenariusz 3: Sprawdzenie czy w raw_downloads już są pobrane dane
    elif (args.raw_dir / "ragwort").exists() and (args.raw_dir / "jkk").exists():
        print(f"[INFO] Wykryto pobrane dane w {args.raw_dir}. Dzielę je na foldery.")
        dataset_dirs = {"ragwort": args.raw_dir / "ragwort", "jkk": args.raw_dir / "jkk"}
        n_rag, n_oth = split_into_classification(dataset_dirs, args.output_dir)

    # Scenariusz 4: Pobieranie przez API Roboflow
    else:
        if not args.api_key:
            raise SystemExit(
                "Błąd: Wymagany jest klucz API Roboflow do pobrania danych.\n"
                "Podaj parametr --api-key TWOJ_KLUCZ lub ustaw zmienną ROBOFLOW_API_KEY,\n"
                "albo wskaż istniejące dane przez --from-combined."
            )
        if not ROBOFLOW_AVAILABLE:
            raise SystemExit("Błąd: Brakuje biblioteki 'roboflow'. Zainstaluj: pip install roboflow")

        rf = Roboflow(api_key=args.api_key)
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        dataset_dirs = download_raw_datasets(rf, args.raw_dir)
        n_rag, n_oth = split_into_classification(dataset_dirs, args.output_dir)

        if not args.keep_raw:
            print(f"[porządki] usuwam surowe dane z {args.raw_dir}")
            shutil.rmtree(args.raw_dir, ignore_errors=True)

    print("\n--- Podsumowanie ---")
    print(f"Zdjęcia ze starcem (ragwort): {n_rag} -> {args.output_dir / 'ragwort'}")
    print(f"Zdjęcia bez starca (others):  {n_oth} -> {args.output_dir / 'others'}")
    print("\nJak wczytać zbiór w PyTorch pod DINOv3:")
    print("  from torchvision.datasets import ImageFolder")
    print(f"  dataset = ImageFolder(r'{args.output_dir}')")
    print("  print(dataset.classes)  # ['others', 'ragwort']")


if __name__ == "__main__":
    main()
