"""
fine_tuning.py

Dwuetapowe doszkolenie (fine-tuning) modelu detekcji starca (RagwortDetection):
  1. Stage 1 — Realne dane:
     - Użycie wyłącznie danych rzeczywistych z data/facility (negatywy z pustymi .txt)
       oraz data/data_concatenated/Felix (pozytywne adnotacje ragwort).
     - Niska wartość learning rate (np. 0.001 z AdamW).
     - Trening przez ~10 epok.
     - Zapis wag i wyników do outputs/fine_labeling_results/stage1.

  2. Stage 2 — Weighted real data:
     - Start od najlepszego modelu z etapu 1.
     - Włączenie ograniczonej puli danych syntetycznych (np. 200-300 próbek).
     - Silne zwagowanie/oversampling około 200 dobrych zdjęć Felixa (~12-15x),
       aby dominowały gradienty w każdej epoce (~75-85% udziału).
     - Kolejne 5-10 epok z niskim learning rate (np. 0.0005).
     - Zapis wag i wyników do outputs/fine_labeling_results/stage2.

Skrypt automatycznie:
  - Analizuje architekturę bazowego modelu best.pth / best.pt (YOLO / DEIM).
  - Przygotowuje manifesty i konfiguracje bez modyfikacji oryginalnych plików danych.
  - Ewaluuje model bazowy, model po etapie 1 i po etapie 2 na wspólnym zbiorze walidacyjnym.
  - Zapisuje metryki porównawcze (JSON, CSV, raport tekstowy) do outputs/fine_labeling_results/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml

# Dodanie katalogu głównego projektu do sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Konfiguracja katalogów wyjściowych Ultralytics
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
FINE_RESULTS_DIR = DEFAULT_OUTPUTS_DIR / "fine_labeling_results"
DEFAULT_DATASETS_DIR = FINE_RESULTS_DIR / "datasets"
STAGE1_DIR = FINE_RESULTS_DIR / "stage1"
STAGE2_DIR = FINE_RESULTS_DIR / "stage2"

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str((DEFAULT_OUTPUTS_DIR / "weights").resolve()),
        "runs_dir": str((FINE_RESULTS_DIR / "runs").resolve()),
    })
except Exception:
    pass

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_model_path(path_str: Optional[Union[str, Path]] = None) -> Path:
    """
    Wyszukuje i normalizuje ścieżkę do wag modelu bazowego (best.pth lub best.pt).
    Zapewnia automatyczne dopasowanie rozszerzenia .pth <-> .pt.
    """
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            candidates = [
                REPO_ROOT / p,
                DEFAULT_OUTPUTS_DIR / p,
                DEFAULT_OUTPUTS_DIR / "best_model" / p.name,
                DEFAULT_OUTPUTS_DIR / "weights" / p.name,
                DEFAULT_OUTPUTS_DIR / "models" / p.name,
            ]
        else:
            candidates = [p]
    else:
        candidates = [
            DEFAULT_OUTPUTS_DIR / "best_model" / "best.pth",
            DEFAULT_OUTPUTS_DIR / "best_model" / "best.pt",
            DEFAULT_OUTPUTS_DIR / "weights" / "best.pt",
            DEFAULT_OUTPUTS_DIR / "models" / "ragwort_yolov8_best.pt",
        ]

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
        # Sprawdź zamianę rozszerzenia .pth <-> .pt
        alt_cand = cand.with_suffix(".pt") if cand.suffix.lower() == ".pth" else cand.with_suffix(".pth")
        if alt_cand.is_file():
            return alt_cand.resolve()

    raise FileNotFoundError(
        f"Nie znaleziono wag modelu bazowego. Przeszukane lokalizacje: {[str(c) for c in candidates]}"
    )


def inspect_model_architecture(model_path: Path) -> Dict[str, Any]:
    """
    Analizuje strukturę pliku wag (state dict / checkpoint) bez założeń wstępnych,
    rozpoznając architekturę (YOLOv8 Ultralytics vs DEIM vs inna).
    """
    print(f"\n[Analiza] Sprawdzanie architektury modelu w pliku: {model_path}")
    try:
        data = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Błąd ładowania pliku wag {model_path}: {e}")

    arch_info: Dict[str, Any] = {
        "framework": "unknown",
        "model_type": None,
        "classes": {0: "ragwort", 1: "objects"},
        "imgsz": 640,
    }

    if isinstance(data, dict):
        if "model" in data and hasattr(data["model"], "yaml"):
            # Ultralytics YOLO model
            arch_info["framework"] = "ultralytics_yolo"
            m = data["model"]
            arch_info["model_type"] = type(m).__name__
            if hasattr(m, "names") and m.names:
                arch_info["classes"] = m.names
            if "train_args" in data and isinstance(data["train_args"], dict):
                arch_info["imgsz"] = data["train_args"].get("imgsz", 640)
            print(f"[Analiza] Rozpoznano model Ultralytics YOLO: {arch_info['model_type']}")
            print(f"[Analiza] Klasy modelu: {arch_info['classes']}, imgsz: {arch_info['imgsz']}")
            return arch_info

        elif "model_state_dict" in data and "config" in data:
            # Model DEIM / DINOv3
            arch_info["framework"] = "deim"
            arch_info["model_type"] = "DEIMModel"
            cfg = data.get("config", {})
            arch_info["classes"] = {c.get("id", 1): c.get("name", "ragwort") for c in data.get("categories", [])}
            print(f"[Analiza] Rozpoznano model DEIM / DINOv3: {arch_info['model_type']}")
            return arch_info

    # Domyślny fallback: próba przez Ultralytics YOLO
    arch_info["framework"] = "ultralytics_yolo"
    arch_info["model_type"] = "YOLO_fallback"
    print(f"[Analiza] Fallback do frameworka Ultralytics YOLO dla: {model_path.name}")
    return arch_info


def discover_facility_images(facility_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Znajduje zdjęcia z data/facility oraz odpowiadające im puste pliki etykiet .txt (negatywy).
    Zwraca listę krotek (ścieżka_obrazu, ścieżka_etykiety).
    """
    if not facility_dir.exists():
        raise FileNotFoundError(f"Katalog data/facility nie istnieje: {facility_dir}")

    samples: List[Tuple[Path, Path]] = []
    for p in sorted(facility_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
            # Szukamy etykiety .txt w tym samym folderze lub w labels/
            lbl_candidates = [
                facility_dir / f"{p.stem}.txt",
                facility_dir / "labels" / f"{p.stem}.txt",
            ]
            lbl_path = None
            for c in lbl_candidates:
                if c.is_file():
                    lbl_path = c
                    break

            if lbl_path is None:
                # Jeśli plik etykiety nie istnieje, wskazujemy domyślną ścieżkę .txt w tym samym folderze
                lbl_path = facility_dir / f"{p.stem}.txt"

            samples.append((p.resolve(), lbl_path.resolve()))

    print(f"[Zbiory] data/facility: znaleziono {len(samples)} zdjęć negatywnych (puste etykiety .txt)")
    return samples


def discover_felix_images(felix_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Znajduje dobre zdjęcia z data/data_concatenated/Felix (lub Felix_data)
    oraz dopasowane etykiety YOLO .txt.
    """
    if not felix_dir.exists():
        # Fallback do Felix_data jeśli Felix nie istnieje
        alt = felix_dir.parent / "Felix_data"
        if alt.exists():
            felix_dir = alt
        else:
            raise FileNotFoundError(f"Katalog Felix nie istnieje: {felix_dir}")

    image_paths: List[Path] = []
    for p in felix_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
            image_paths.append(p.resolve())

    samples: List[Tuple[Path, Path]] = []
    for img in sorted(image_paths):
        # Sprawdzamy możliwe lokalizacje etykiety YOLO
        stem = img.stem
        parent = img.parent
        possible_lbls = [
            parent / f"{stem}.txt",
            parent.parent / "annotation" / f"{stem}.txt",
            parent.parent / "labels" / f"{stem}.txt",
            parent / "labels" / f"{stem}.txt",
        ]
        lbl = None
        for cand in possible_lbls:
            if cand.is_file():
                lbl = cand
                break

        if lbl is None:
            lbl = parent / f"{stem}.txt"

        samples.append((img, lbl.resolve()))

    print(f"[Zbiory] Felix ({felix_dir.name}): znaleziono {len(samples)} dobrych zdjęć z adnotacjami")
    return samples


def discover_synthetic_images(
    synth_dir: Optional[Path] = None,
    limit: int = 300,
) -> List[Tuple[Path, Path]]:
    """
    Wyszukuje ograniczoną pulę zdjęć syntetycznych wraz z ich etykietami.
    """
    candidates = [
        synth_dir,
        REPO_ROOT / "data" / "data_concatenated" / "synthetic" / "images",
        REPO_ROOT / "data" / "data_concatenated" / "synthetic_dataset_split" / "part_1" / "images",
        REPO_ROOT / "data" / "synthetic_dataset" / "images",
    ]

    chosen_dir: Optional[Path] = None
    for cand in candidates:
        if cand and cand.is_dir() and any(cand.iterdir()):
            chosen_dir = cand
            break

    if chosen_dir is None:
        print("[Zbiory] Ostrzeżenie: Nie znaleziono katalogu z danymi syntetycznymi. Pomijam syntetyki.")
        return []

    image_paths = [p for p in chosen_dir.glob("*.jpg") if p.is_file()]
    if not image_paths:
        image_paths = [p for p in chosen_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS]

    # Deterministic shuffle
    rnd = random.Random(42)
    rnd.shuffle(image_paths)
    selected = image_paths[:limit]

    samples: List[Tuple[Path, Path]] = []
    for img in selected:
        stem = img.stem
        # YOLO labels location
        cand_lbls = [
            img.parent.parent / "labels" / f"{stem}.txt",
            img.parent / f"{stem}.txt",
        ]
        lbl = None
        for cl in cand_lbls:
            if cl.is_file():
                lbl = cl
                break
        if lbl is None:
            lbl = img.parent.parent / "labels" / f"{stem}.txt"
        samples.append((img.resolve(), lbl.resolve()))

    print(f"[Zbiory] Dane syntetyczne ({chosen_dir.name}): wybrano {len(samples)} zdjęć (limit: {limit})")
    return samples


def discover_validation_images(
    val_felix: List[Tuple[Path, Path]],
    val_facility: List[Tuple[Path, Path]],
) -> List[Path]:
    """
    Łączy próbkę walidacyjną:
      - wydzielone zdjęcia Felix (pozytywy),
      - wydzielone zdjęcia facility (negatywy),
      - oraz zdjęcia z istniejącego zbioru testowego combined_dataset.
    """
    val_paths: List[Path] = [p for p, _ in val_felix] + [p for p, _ in val_facility]

    test_dir_candidates = [
        REPO_ROOT / "data" / "data_concatenated" / "combined_dataset" / "test" / "images",
        REPO_ROOT / "data" / "combined_dataset" / "test" / "images",
    ]
    for td in test_dir_candidates:
        if td.is_dir():
            for p in sorted(td.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS:
                    val_paths.append(p.resolve())
            break

    return val_paths


def prepare_fine_tuning_datasets(
    facility_dir: Path,
    felix_dir: Path,
    output_datasets_dir: Path,
    synth_dir: Optional[Path] = None,
    felix_weight_stage2: int = 12,
    facility_weight_stage2: int = 1,
    synth_limit_stage2: int = 250,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Automatycznie przygotowuje manifesty i konfiguracje YAML dla obu etapów.
    Zapewnia pełną bezinwazyjność (brak modyfikacji plików źródłowych).
    """
    output_datasets_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)

    print("\n" + "=" * 78)
    print(" PRZYGOTOWANIE DATASETÓW DO FINE-TUNINGU (BEZINWAZYJNE)")
    print("=" * 78)

    # 1. Pozyskanie zdjęć realnych
    facility_samples = discover_facility_images(facility_dir)
    felix_samples = discover_felix_images(felix_dir)

    # 2. Wydzielenie części walidacyjnej z danych realnych
    rnd.shuffle(facility_samples)
    rnd.shuffle(felix_samples)

    n_val_felix = max(1, int(len(felix_samples) * val_ratio))
    val_felix = felix_samples[:n_val_felix]
    train_felix = felix_samples[n_val_felix:]

    n_val_facility = max(1, int(len(facility_samples) * val_ratio))
    val_facility = facility_samples[:n_val_facility]
    train_facility = facility_samples[n_val_facility:]

    # 3. Zbiór walidacyjny wspólny dla obu etapów i baseline'u
    val_images = discover_validation_images(val_felix, val_facility)

    # 4. Dane syntetyczne dla etapu 2
    synth_samples = discover_synthetic_images(synth_dir, limit=synth_limit_stage2)

    # 5. Generowanie manifestów YOLO (.txt)
    def to_line(p: Path) -> str:
        # Zapisujemy ścieżki relatywne względem REPO_ROOT
        try:
            rel = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            return rel
        except ValueError:
            return str(p.resolve())

    # --- Manifest Stage 1: czyste dane realne (Felix + Facility) ---
    stage1_lines: List[str] = []
    for img, _ in train_felix:
        stage1_lines.append(to_line(img))
    for img, _ in train_facility:
        stage1_lines.append(to_line(img))
    rnd.shuffle(stage1_lines)

    # --- Manifest Stage 2: weighted real data + ograniczony syntetyk ---
    stage2_lines: List[str] = []
    # Zwagowane próbkowanie zdjęć Felixa
    for img, _ in train_felix:
        line = to_line(img)
        for _ in range(felix_weight_stage2):
            stage2_lines.append(line)

    # Zdjęcia facility (negatywy)
    for img, _ in train_facility:
        line = to_line(img)
        for _ in range(facility_weight_stage2):
            stage2_lines.append(line)

    # Ograniczony udział danych syntetycznych
    for img, _ in synth_samples:
        stage2_lines.append(to_line(img))

    rnd.shuffle(stage2_lines)

    # --- Manifest Walidacji ---
    val_lines = [to_line(p) for p in val_images]

    # Zapis plików manifestów
    stage1_txt = output_datasets_dir / "stage1_train.txt"
    stage2_txt = output_datasets_dir / "stage2_train.txt"
    val_txt = output_datasets_dir / "val.txt"

    stage1_txt.write_text("\n".join(stage1_lines) + "\n", encoding="utf-8")
    stage2_txt.write_text("\n".join(stage2_lines) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    # Zapis plików konfiguracji data.yaml dla YOLO
    base_data_cfg = {
        "path": str(REPO_ROOT.resolve()),
        "val": str(val_txt.relative_to(REPO_ROOT).as_posix()),
        "test": str(val_txt.relative_to(REPO_ROOT).as_posix()),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }

    cfg_s1 = dict(base_data_cfg)
    cfg_s1["train"] = str(stage1_txt.relative_to(REPO_ROOT).as_posix())
    stage1_yaml = output_datasets_dir / "stage1_data.yaml"
    with open(stage1_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg_s1, f, sort_keys=False, allow_unicode=True)

    cfg_s2 = dict(base_data_cfg)
    cfg_s2["train"] = str(stage2_txt.relative_to(REPO_ROOT).as_posix())
    stage2_yaml = output_datasets_dir / "stage2_data.yaml"
    with open(stage2_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg_s2, f, sort_keys=False, allow_unicode=True)

    # Statystyki i proporcje
    n_s2_felix = len(train_felix) * felix_weight_stage2
    n_s2_facility = len(train_facility) * facility_weight_stage2
    n_s2_synth = len(synth_samples)
    total_s2 = len(stage2_lines)
    pct_felix_s2 = (n_s2_felix / total_s2 * 100) if total_s2 > 0 else 0
    pct_synth_s2 = (n_s2_synth / total_s2 * 100) if total_s2 > 0 else 0

    print(f"\n[Konfiguracja Stage 1 - Realne Dane]")
    print(f"  - Pozytywne (Felix):        {len(train_felix)} zdjęć")
    print(f"  - Negatywne (Facility):     {len(train_facility)} zdjęć")
    print(f"  - Łącznie w epoce:          {len(stage1_lines)} kroków")
    print(f"  - Plik konfiguracji:        {stage1_yaml}")

    print(f"\n[Konfiguracja Stage 2 - Weighted Real Data]")
    print(f"  - Pozytywne (Felix {felix_weight_stage2}x):   {len(train_felix)} unikalnych -> {n_s2_felix} próbek ({pct_felix_s2:.1f}% udziału)")
    print(f"  - Negatywne (Facility {facility_weight_stage2}x): {len(train_facility)} unikalnych -> {n_s2_facility} próbek")
    print(f"  - Syntetyczne (1x):         {n_s2_synth} próbek ({pct_synth_s2:.1f}% udziału)")
    print(f"  - Łącznie w epoce:          {total_s2} kroków")
    print(f"  - Plik konfiguracji:        {stage2_yaml}")

    print(f"\n[Zbiór Walidacyjny]")
    print(f"  - Liczba próbek testowych:  {len(val_lines)} zdjęć")
    print(f"  - Manifest walidacji:       {val_txt}")
    print("=" * 78)

    return {
        "stage1_yaml": stage1_yaml,
        "stage2_yaml": stage2_yaml,
        "val_txt": val_txt,
        "stage1_txt": stage1_txt,
        "stage2_txt": stage2_txt,
    }


def evaluate_checkpoint(
    model_weight: Union[str, Path],
    data_yaml: Path,
    device: str,
    imgsz: int = 640,
    batch: int = 16,
) -> Dict[str, float]:
    """
    Przeprowadza walidację zadanego checkpointu i zwraca ujednolicony słownik metryk.
    """
    from ultralytics import YOLO

    weight_p = Path(model_weight)
    if not weight_p.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku wag do ewaluacji: {weight_p}")

    print(f"[Walidacja] Ładowanie wag: {weight_p.name} na {device}...")
    model = YOLO(str(weight_p), task="detect")
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        split="val",
        device=device,
        verbose=False,
    )

    metrics = {
        "mAP50": float(res.box.map50) if hasattr(res, "box") else 0.0,
        "mAP50-95": float(res.box.map) if hasattr(res, "box") else 0.0,
        "precision": float(res.box.mp) if hasattr(res, "box") else 0.0,
        "recall": float(res.box.mr) if hasattr(res, "box") else 0.0,
        "fitness": float(res.fitness) if hasattr(res, "fitness") else 0.0,
    }
    return metrics


class FineTuningPipeline:
    """
    Dwustopniowy potok doszkalania modelu:
      Baseline -> Stage 1 (Real) -> Stage 2 (Weighted Real).
    """

    def __init__(
        self,
        baseline_model_path: Optional[Union[str, Path]] = None,
        facility_dir: Optional[Union[str, Path]] = None,
        felix_dir: Optional[Union[str, Path]] = None,
        synth_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch: int = 16,
        imgsz: int = 640,
        workers: int = 4,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.baseline_path = resolve_model_path(baseline_model_path)
        self.facility_dir = Path(facility_dir or (REPO_ROOT / "data" / "facility")).resolve()
        self.felix_dir = Path(felix_dir or (REPO_ROOT / "data" / "data_concatenated" / "Felix")).resolve()
        self.synth_dir = Path(synth_dir).resolve() if synth_dir else None
        self.output_dir = Path(output_dir or FINE_RESULTS_DIR).resolve()
        self.stage1_dir = self.output_dir / "stage1"
        self.stage2_dir = self.output_dir / "stage2"
        self.datasets_dir = self.output_dir / "datasets"

        self.batch = batch
        self.imgsz = imgsz
        self.workers = workers

        self.metrics_summary: Dict[str, Dict[str, Any]] = {}

    def prepare_data(
        self,
        felix_weight_stage2: int = 12,
        synth_limit_stage2: int = 250,
    ) -> Dict[str, Path]:
        """Przygotowuje datasety do Stage 1 i Stage 2."""
        return prepare_fine_tuning_datasets(
            facility_dir=self.facility_dir,
            felix_dir=self.felix_dir,
            output_datasets_dir=self.datasets_dir,
            synth_dir=self.synth_dir,
            felix_weight_stage2=felix_weight_stage2,
            synth_limit_stage2=synth_limit_stage2,
        )

    def evaluate_baseline(self, val_yaml: Path) -> Dict[str, float]:
        """Ocenia model bazowy (baseline) przed fine-tuningiem."""
        print("\n" + "=" * 78)
        print(" 1. EWALUACJA MODELU BAZOWEGO (BASELINE)")
        print(f" Checkpoint: {self.baseline_path}")
        print("=" * 78)

        metrics = evaluate_checkpoint(
            model_weight=self.baseline_path,
            data_yaml=val_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )

        self.metrics_summary["baseline"] = {
            "checkpoint": str(self.baseline_path),
            "metrics": metrics,
        }
        self._print_metrics_block("BASELINE", metrics)
        return metrics

    def run_stage1(
        self,
        data_yaml: Path,
        epochs: int = 10,
        lr0: float = 0.001,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Etap 1: Doszkalanie na czystych danych rzeczywistych (facility + felix).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" 2. FINE-TUNING ETAP 1 — REALNE DANE")
        print(f" Model startowy:   {self.baseline_path}")
        print(f" Epoki:            {epochs}")
        print(f" Learning Rate:    lr0={lr0}, lrf={lrf}")
        print(f" Optymalizator:    {optimizer}")
        print(f" Dane:             {data_yaml}")
        print(f" Katalog wyjścia:  {self.stage1_dir}")
        print("=" * 78)

        self.stage1_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self.baseline_path), task="detect")

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage1_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage1_dir / "run" / "weights" / "best.pt"
        if not best_trained.is_file():
            # Fallback jeśli Ultralytics zapisał w podkatalogu weights bezpośrednio
            candidates = list(self.stage1_dir.rglob("best.pt"))
            if candidates:
                best_trained = candidates[0]
            else:
                raise FileNotFoundError(f"Nie znaleziono pliku best.pt po treningu Stage 1 w: {self.stage1_dir}")

        # Zapisz czytelne kopie best.pt oraz best.pth w głównym folderze stage1
        stage1_best_pt = self.stage1_dir / "best.pt"
        stage1_best_pth = self.stage1_dir / "best.pth"
        shutil.copy2(best_trained, stage1_best_pt)
        try:
            if stage1_best_pth.exists() or stage1_best_pth.is_symlink():
                stage1_best_pth.unlink()
            stage1_best_pth.symlink_to(stage1_best_pt.name)
        except Exception:
            shutil.copy2(best_trained, stage1_best_pth)

        print(f"\n[Stage 1] Zapisano najlepszy model: {stage1_best_pt} oraz {stage1_best_pth}")

        # Ewaluacja checkpointu Stage 1
        metrics = evaluate_checkpoint(
            model_weight=stage1_best_pt,
            data_yaml=data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage1"] = {
            "checkpoint": str(stage1_best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 1 (REAL DATA)", metrics)
        return stage1_best_pt

    def run_stage2(
        self,
        stage1_weight: Path,
        data_yaml: Path,
        epochs: int = 10,
        lr0: float = 0.0005,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Etap 2: Doszkalanie ze zwagowanymi danymi Felixa i ograniczonym udziałem syntetyków.
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" 3. FINE-TUNING ETAP 2 — WEIGHTED REAL DATA")
        print(f" Model startowy:   {stage1_weight}")
        print(f" Epoki:            {epochs}")
        print(f" Learning Rate:    lr0={lr0}, lrf={lrf}")
        print(f" Optymalizator:    {optimizer}")
        print(f" Dane:             {data_yaml}")
        print(f" Katalog wyjścia:  {self.stage2_dir}")
        print("=" * 78)

        self.stage2_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(stage1_weight), task="detect")

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage2_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage2_dir / "run" / "weights" / "best.pt"
        if not best_trained.is_file():
            candidates = list(self.stage2_dir.rglob("best.pt"))
            if candidates:
                best_trained = candidates[0]
            else:
                raise FileNotFoundError(f"Nie znaleziono pliku best.pt po treningu Stage 2 w: {self.stage2_dir}")

        stage2_best_pt = self.stage2_dir / "best.pt"
        stage2_best_pth = self.stage2_dir / "best.pth"
        shutil.copy2(best_trained, stage2_best_pt)
        try:
            if stage2_best_pth.exists() or stage2_best_pth.is_symlink():
                stage2_best_pth.unlink()
            stage2_best_pth.symlink_to(stage2_best_pt.name)
        except Exception:
            shutil.copy2(best_trained, stage2_best_pth)

        print(f"\n[Stage 2] Zapisano finalny model: {stage2_best_pt} oraz {stage2_best_pth}")

        # Kopia do outputs/models/ dla łatwej integracji z resztą repozytorium
        try:
            models_dir = DEFAULT_OUTPUTS_DIR / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage2_best_pt, models_dir / "ragwort_finetuned_stage2_best.pt")
        except Exception:
            pass

        # Ewaluacja checkpointu Stage 2
        metrics = evaluate_checkpoint(
            model_weight=stage2_best_pt,
            data_yaml=data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage2"] = {
            "checkpoint": str(stage2_best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 2 (WEIGHTED REAL DATA)", metrics)
        return stage2_best_pt

    def save_comparison_report(self):
        """Generuje raporty porównawcze baseline -> stage1 -> stage2."""
        json_path = self.output_dir / "metrics_comparison.json"
        csv_path = self.output_dir / "metrics_comparison.csv"
        txt_path = self.output_dir / "summary.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)

        # Tworzenie pliku CSV
        csv_lines = ["stage,checkpoint,mAP50,mAP50-95,precision,recall,fitness"]
        for st_name in ["baseline", "stage1", "stage2"]:
            if st_name in self.metrics_summary:
                info = self.metrics_summary[st_name]
                m = info["metrics"]
                csv_lines.append(
                    f"{st_name},{info['checkpoint']},{m.get('mAP50', 0):.4f},"
                    f"{m.get('mAP50-95', 0):.4f},{m.get('precision', 0):.4f},"
                    f"{m.get('recall', 0):.4f},{m.get('fitness', 0):.4f}"
                )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        # Tworzenie czytelnego raportu tekstowego i wydruku do konsoli
        header = "=" * 82 + "\n"
        title = "             PORÓWNANIE METRYK: BASELINE -> STAGE 1 -> STAGE 2\n"
        sep = "-" * 82 + "\n"
        col_names = f"{'Etap':<12} | {'mAP@50':<10} | {'mAP@50-95':<12} | {'Precyzja':<10} | {'Czułość':<10} | {'Fitness':<10}\n"

        rows = []
        for st_name in ["baseline", "stage1", "stage2"]:
            if st_name in self.metrics_summary:
                m = self.metrics_summary[st_name]["metrics"]
                label = "Baseline" if st_name == "baseline" else ("Stage 1" if st_name == "stage1" else "Stage 2")
                row = (
                    f"{label:<12} | "
                    f"{m.get('mAP50', 0):<10.4f} | "
                    f"{m.get('mAP50-95', 0):<12.4f} | "
                    f"{m.get('precision', 0):<10.4f} | "
                    f"{m.get('recall', 0):<10.4f} | "
                    f"{m.get('fitness', 0):<10.4f}\n"
                )
                rows.append(row)

        report_txt = header + title + header + col_names + sep + "".join(rows) + header
        txt_path.write_text(report_txt, encoding="utf-8")

        print("\n" + report_txt)
        print(f"[Raport] Wyniki zapisano w:")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV:  {csv_path}")
        print(f"  - TXT:  {txt_path}")

    def _print_metrics_block(self, title: str, metrics: Dict[str, float]):
        print(f"\n--- WYNIKI WALIDACJI: {title} ---")
        print(f"  mAP@50               : {metrics.get('mAP50', 0.0):.4f}")
        print(f"  mAP@50-95            : {metrics.get('mAP50-95', 0.0):.4f}")
        print(f"  Precyzja (Precision) : {metrics.get('precision', 0.0):.4f}")
        print(f"  Czułość (Recall)     : {metrics.get('recall', 0.0):.4f}")
        print(f"  Fitness              : {metrics.get('fitness', 0.0):.4f}")
        print("---------------------------------------")


def run_fine_tuning(
    model_path: Optional[Union[str, Path]] = None,
    facility_dir: Optional[Union[str, Path]] = None,
    felix_dir: Optional[Union[str, Path]] = None,
    synth_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    stage1_epochs: int = 10,
    stage2_epochs: int = 10,
    lr0_stage1: float = 0.001,
    lr0_stage2: float = 0.0005,
    felix_weight_stage2: int = 12,
    synth_limit_stage2: int = 250,
    batch: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None,
    workers: int = 4,
    skip_stage1: bool = False,
    stage1_checkpoint: Optional[Union[str, Path]] = None,
    val_only: bool = False,
):
    """
    Główna funkcja uruchamiająca cały dwuetapowy proces fine-tuningu.
    """
    # 1. Sprawdzenie ścieżki i analiza architektury modelu
    resolved_model = resolve_model_path(model_path)
    arch_info = inspect_model_architecture(resolved_model)

    pipeline = FineTuningPipeline(
        baseline_model_path=resolved_model,
        facility_dir=facility_dir,
        felix_dir=felix_dir,
        synth_dir=synth_dir,
        output_dir=output_dir,
        device=device,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
    )

    # 2. Przygotowanie datasetów
    datasets = pipeline.prepare_data(
        felix_weight_stage2=felix_weight_stage2,
        synth_limit_stage2=synth_limit_stage2,
    )

    # 3. Ewaluacja bazowa
    pipeline.evaluate_baseline(val_yaml=datasets["stage1_yaml"])
    if val_only:
        print("[Info] Uruchomiono w trybie --val-only. Zakończono po ewaluacji bazowej.")
        return pipeline.metrics_summary

    # 4. Stage 1 — Realne dane
    if skip_stage1:
        if stage1_checkpoint and Path(stage1_checkpoint).is_file():
            best_s1 = Path(stage1_checkpoint).resolve()
        else:
            best_s1 = pipeline.stage1_dir / "best.pt"
            if not best_s1.is_file():
                raise FileNotFoundError(f"Flaga --skip-stage1 aktywna, ale nie znaleziono wag w: {best_s1}")
        print(f"\n[Info] Pomijam trening Stage 1. Używam istniejącego checkpointu: {best_s1}")
        metrics_s1 = evaluate_checkpoint(
            best_s1, datasets["stage1_yaml"], device=pipeline.device, imgsz=imgsz, batch=batch
        )
        pipeline.metrics_summary["stage1"] = {"checkpoint": str(best_s1), "metrics": metrics_s1}
    else:
        best_s1 = pipeline.run_stage1(
            data_yaml=datasets["stage1_yaml"],
            epochs=stage1_epochs,
            lr0=lr0_stage1,
        )

    # 5. Stage 2 — Weighted real data
    best_s2 = pipeline.run_stage2(
        stage1_weight=best_s1,
        data_yaml=datasets["stage2_yaml"],
        epochs=stage2_epochs,
        lr0=lr0_stage2,
    )

    # 6. Raport porównawczy (Baseline -> Stage 1 -> Stage 2)
    pipeline.save_comparison_report()
    return pipeline.metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dwuetapowe doszkalanie (fine-tuning) modelu detekcji starca (RagwortDetection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/best_model/best.pth",
        help="Ścieżka do wag modelu bazowego (np. best.pth lub best.pt)",
    )
    parser.add_argument(
        "--facility-dir",
        type=Path,
        default=REPO_ROOT / "data" / "facility",
        help="Ścieżka do katalogu ze zdjęciami z obiektu (negatywy z pustymi .txt)",
    )
    parser.add_argument(
        "--felix-dir",
        type=Path,
        default=REPO_ROOT / "data" / "data_concatenated" / "Felix",
        help="Ścieżka do dobrych zdjęć Felixa",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=None,
        help="Katalog ze zdjęciami syntetycznymi (domyślnie auto-wykrywanie)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINE_RESULTS_DIR,
        help="Główny katalog wyników doszkalania",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=10,
        help="Liczba epok dla etapu 1 (realne dane)",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=10,
        help="Liczba epok dla etapu 2 (weighted real data)",
    )
    parser.add_argument(
        "--lr0-stage1",
        type=float,
        default=0.001,
        help="Początkowy learning rate dla etapu 1",
    )
    parser.add_argument(
        "--lr0-stage2",
        type=float,
        default=0.0005,
        help="Początkowy learning rate dla etapu 2",
    )
    parser.add_argument(
        "--felix-weight",
        type=int,
        default=12,
        help="Współczynnik próbkowania/waga dla zdjęć Felixa w etapie 2",
    )
    parser.add_argument(
        "--synth-limit",
        type=int,
        default=250,
        help="Maksymalna liczba zdjęć syntetycznych włączanych w etapie 2",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Rozmiar batcha treningowego",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Rozdzielczość obrazu wejściowego",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Liczba procesów roboczych DataLoader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Urządzenie obliczeniowe ('0', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Pomiń trening etapu 1 i przejdź do etapu 2 z istniejącym checkpointem",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Ścieżka do checkpointu etapu 1 (gdy używamy --skip-stage1)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Wykonaj tylko ewaluację modelu bazowego bez przeprowadzania treningu",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_fine_tuning(
        model_path=args.model,
        facility_dir=args.facility_dir,
        felix_dir=args.felix_dir,
        synth_dir=args.synth_dir,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lr0_stage1=args.lr0_stage1,
        lr0_stage2=args.lr0_stage2,
        felix_weight_stage2=args.felix_weight,
        synth_limit_stage2=args.synth_limit,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        skip_stage1=args.skip_stage1,
        stage1_checkpoint=args.stage1_checkpoint,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()