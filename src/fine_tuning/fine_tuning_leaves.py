"""
fine_tuning_leaves.py

Rozszerzenie pipeline'u fine-tuningu o kolejne etapy (Stages 3, 4 i 5):
Ukierunkowane na naukę cech strukturalnych, liści i geometrii starca jakubka (Ragwort),
zamiast polegania wyłącznie na żółtym kolorze kwiatów.

Etapy:
  1. Stage 3 — Zwiększona rozdzielczość:
     - Start od checkpointu z Stage 2 (outputs/fine_labeling_results/stage2/best.pt).
     - Użycie danych zwagowanych z etapu 2.
     - Zwiększenie rozdzielczości do imgsz=1024.
     - Niski learning rate (lr0=0.0005), optymalizator AdamW.
     - Zapis do outputs/fine_labeling_results/leaves_stage3_resolution.

  2. Stage 4 — Wymuszenie nauki cech liści (Augmentacja):
     - Start od najlepszego modelu z Stage 3.
     - Włączenie mocnych augmentacji:
       erasing=0.4 (losowe wymazywanie/cutout, przesłanianie kwiatów),
       hsv_s=0.8 (drastyczna zmiana nasycenia barw),
       hsv_v=0.5 (duża zmienność jasności),
       imgsz=1024.
     - Niski learning rate (lr0=0.0003).
     - Zapis do outputs/fine_labeling_results/leaves_stage4_augmentation.

  3. Stage 5 — Trening na zdjęciach bez koloru żółtego (No-Yellow):
     - Start od najlepszego modelu z Stage 4.
     - Utworzenie kopii danych treningowych (oryginalne dane pozostają nietknięte).
     - Zastosowanie kontrolowanej transformacji w przestrzeni HSV:
       desaturacja pikseli w zakresie żółci (H in [18, 36]) do neutralnych odcieni,
       przy zachowaniu naturalnych zielonych barw liści (H in [38, 85]) oraz geometrii/etykiet.
     - Trening na imgsz=1024 z niskim learning rate (lr0=0.0003).
     - Zapis do outputs/fine_labeling_results/leaves_stage5_no_yellow.

Podsumowanie:
  Zestawienie metryk Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 na wspólnym zbiorze walidacyjnym.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

# Dodanie katalogu głównego do sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Konfiguracja katalogów Ultralytics
FINE_RESULTS_DIR = REPO_ROOT / "outputs" / "fine_labeling_results"
STAGE2_DEFAULT_MODEL = FINE_RESULTS_DIR / "stage2" / "best.pt"
DATASETS_DIR = FINE_RESULTS_DIR / "datasets"

STAGE3_DIR = FINE_RESULTS_DIR / "leaves_stage3_resolution"
STAGE4_DIR = FINE_RESULTS_DIR / "leaves_stage4_augmentation"
STAGE5_DIR = FINE_RESULTS_DIR / "leaves_stage5_no_yellow"

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str((REPO_ROOT / "outputs" / "weights").resolve()),
        "runs_dir": str((FINE_RESULTS_DIR / "runs").resolve()),
    })
except Exception:
    pass

from src.fine_tuning.fine_tuning import (
    evaluate_checkpoint,
    inspect_model_architecture,
    resolve_model_path,
)


def remove_yellow_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Wykonuje kontrolowaną modyfikację kolorystyczną w przestrzeni barw HSV:
    usuwa/desaturuje charakterystyczny żółty kolor kwiatów,
    pozostawiając naturalne odcienie zielonych liści i tła nienaruszone.

    Parametry HSV w OpenCV:
      - Żółty (kwiaty starca): H in [18, 36], S >= 40, V >= 50
      - Zielony (liście i łodygi): H in [38, 85] (brak nakładania się masek)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 40, 50], dtype=np.uint8)
    upper_yellow = np.array([36, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Desaturacja żółtych pikseli (kwiaty stają się neutralnie białawo-szare jak po przekwitnieniu)
    h, s, v = cv2.split(hsv)
    s_modified = np.where(yellow_mask > 0, (s * 0.05).astype(np.uint8), s)
    modified_hsv = cv2.merge([h, s_modified, v])

    return cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)


def prepare_no_yellow_dataset(
    source_manifest: Path,
    output_dataset_dir: Path,
    val_manifest: Path,
) -> Path:
    """
    Przygotowuje kopię danych treningowych z usuniętym kolorem żółtym.
    Oryginalne dane w data/ pozostają całkowicie nienaruszone.
    """
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dataset_dir / "images"
    labels_out = output_dataset_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(" PRZYGOTOWANIE DATASETU NO-YELLOW (ETAP 5)")
    print(f" Manifest źródłowy: {source_manifest}")
    print(f" Katalog docelowy:  {output_dataset_dir}")
    print("=" * 78)

    if not source_manifest.is_file():
        raise FileNotFoundError(f"Brak pliku manifestu źródłowego: {source_manifest}")

    lines = source_manifest.read_text(encoding="utf-8").splitlines()
    unique_image_paths = sorted(list(set(l.strip() for l in lines if l.strip())))

    print(f"[No-Yellow] Przetwarzanie {len(unique_image_paths)} unikalnych zdjęć...")

    rel_to_transformed: Dict[str, str] = {}
    converted_count = 0

    for idx, line_path in enumerate(unique_image_paths):
        p = Path(line_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()

        if not p.is_file():
            continue

        # Nowa unikalna nazwa pliku w no-yellow dataset
        out_img_name = f"noyellow_{p.stem}{p.suffix.lower()}"
        out_img_path = images_out / out_img_name
        out_lbl_path = labels_out / f"{out_img_name.rsplit('.', 1)[0]}.txt"

        # Kopiowanie i modyfikacja obrazu tylko jeśli jeszcze nie istnieje
        if not out_img_path.is_file():
            img_bgr = cv2.imread(str(p))
            if img_bgr is not None:
                transformed_bgr = remove_yellow_from_image(img_bgr)
                cv2.imwrite(str(out_img_path), transformed_bgr)
                converted_count += 1
            else:
                shutil.copy2(p, out_img_path)

        # Dopasowanie etykiety .txt
        if not out_lbl_path.is_file():
            # Szukanie etykiety w oryginalnych lokalizacjach
            stem = p.stem
            cand_lbls = [
                p.parent / f"{stem}.txt",
                p.parent.parent / "labels" / f"{stem}.txt",
                p.parent.parent / "annotation" / f"{stem}.txt",
                p.parent / "labels" / f"{stem}.txt",
            ]
            src_lbl = None
            for c in cand_lbls:
                if c.is_file():
                    src_lbl = c
                    break

            if src_lbl and src_lbl.is_file() and src_lbl.stat().st_size > 0:
                shutil.copy2(src_lbl, out_lbl_path)
            else:
                out_lbl_path.write_text("", encoding="utf-8")

        # Zapisz relatywną ścieżkę do nowego manifestu
        try:
            rel_to_repo = out_img_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel_to_repo = str(out_img_path.resolve())
        rel_to_transformed[line_path] = rel_to_repo

    print(f"[No-Yellow] Zmodyfikowano {converted_count} zdjęć (usunięto żółty kolor kwiatów).")
    print(f"[No-Yellow] Etykiety i geometria zachowane 1:1.")

    # Budowa nowego manifestu treningowego ze zbalansowanymi wagami z source_manifest
    new_train_lines = []
    for l in lines:
        raw_l = l.strip()
        if raw_l in rel_to_transformed:
            new_train_lines.append(rel_to_transformed[raw_l])

    train_no_yellow_txt = output_dataset_dir / "train_no_yellow.txt"
    train_no_yellow_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")

    # Tworzenie pliku data.yaml
    no_yellow_yaml = output_dataset_dir / "data_no_yellow.yaml"
    def get_rel_str(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT.resolve()).as_posix())
        except ValueError:
            return str(p.resolve())

    cfg = {
        "path": str(REPO_ROOT.resolve()),
        "train": get_rel_str(train_no_yellow_txt),
        "val": get_rel_str(val_manifest),
        "test": get_rel_str(val_manifest),
        "nc": 2,
        "names": {0: "ragwort", 1: "objects"},
    }
    with open(no_yellow_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[No-Yellow] Manifest zapisano: {train_no_yellow_txt}")
    print(f"[No-Yellow] Plik YAML:          {no_yellow_yaml}")
    print("=" * 78)

    return no_yellow_yaml


class LeavesFineTuningPipeline:
    """
    Rozszerzony potok doszkalania modelu skupiony na cechach liści i struktury:
      Stage 2 (Base) -> Stage 3 (1024 Res) -> Stage 4 (Augmentations) -> Stage 5 (No-Yellow)
    """

    def __init__(
        self,
        stage2_model_path: Optional[Union[str, Path]] = None,
        data_yaml: Optional[Union[str, Path]] = None,
        val_manifest: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        imgsz: int = 1024,
        batch: int = 8,
        device: Optional[str] = None,
        workers: int = 4,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir or FINE_RESULTS_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Wyszukanie modelu po Stage 2
        cand_stage2 = stage2_model_path or STAGE2_DEFAULT_MODEL
        self.stage2_model_path = resolve_model_path(cand_stage2)

        # Wyszukanie konfiguracji danych z poprzedniego etapu
        default_yaml = DATASETS_DIR / "stage2_data.yaml"
        self.data_yaml = Path(data_yaml or default_yaml).resolve()
        if not self.data_yaml.is_file():
            # Fallback
            cands = list(DATASETS_DIR.glob("*data*.yaml"))
            if cands:
                self.data_yaml = cands[0].resolve()
            else:
                raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji danych YAML: {self.data_yaml}")

        default_val = DATASETS_DIR / "val.txt"
        self.val_manifest = Path(val_manifest or default_val).resolve()

        self.imgsz = imgsz
        self.batch = batch
        self.workers = workers

        self.stage3_dir = self.output_dir / "leaves_stage3_resolution"
        self.stage4_dir = self.output_dir / "leaves_stage4_augmentation"
        self.stage5_dir = self.output_dir / "leaves_stage5_no_yellow"

        self.metrics_summary: Dict[str, Dict[str, Any]] = {}

    def evaluate_initial_stage2(self) -> Dict[str, float]:
        """Ewaluuje model wejściowy ze Stage 2 na rozdzielczości 1024."""
        print("\n" + "=" * 78)
        print(" 0. EWALUACJA WYJŚCIOWA MODELU ZE STAGE 2")
        print(f" Checkpoint: {self.stage2_model_path}")
        print(f" Imgsz:      {self.imgsz}")
        print("=" * 78)

        metrics = evaluate_checkpoint(
            model_weight=self.stage2_model_path,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )

        self.metrics_summary["stage2"] = {
            "checkpoint": str(self.stage2_model_path),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 2 (WYJŚCIOWY)", metrics)
        return metrics

    def run_stage3(
        self,
        epochs: int = 8,
        lr0: float = 0.0005,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 3: Zwiększona rozdzielczość (imgsz=1024).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 3 — ZWIĘKSZONA ROZDZIELCZOŚĆ (imgsz=1024)")
        print(f" Model wejściowy: {self.stage2_model_path}")
        print(f" Epoki:           {epochs}")
        print(f" Batch / Imgsz:   {self.batch} / {self.imgsz}")
        print(f" Learning Rate:   lr0={lr0}, lrf={lrf}")
        print(f" Katalog wyjścia: {self.stage3_dir}")
        print("=" * 78)

        self.stage3_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self.stage2_model_path), task="detect")

        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage3_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage3_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage3_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage3_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku wag best.pt w: {self.stage3_dir}")

        best_pt = self.stage3_dir / "best.pt"
        best_pth = self.stage3_dir / "best.pth"
        last_pt = self.stage3_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 3] Zapisano checkpointy: {best_pt} oraz {last_pt}")

        # Ewaluacja
        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage3"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 3 (RESOLUTION 1024)", metrics)
        return best_pt

    def run_stage4(
        self,
        stage3_weight: Path,
        epochs: int = 8,
        lr0: float = 0.0003,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
        erasing: float = 0.4,
        hsv_s: float = 0.8,
        hsv_v: float = 0.5,
    ) -> Path:
        """
        Stage 4: Wymuszenie nauki cech liści (silniejsze augmentacje erasing i HSV).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 4 — AUGMENTACJE I NAUKA CECH LIŚCI")
        print(f" Model wejściowy: {stage3_weight}")
        print(f" Epoki:           {epochs}")
        print(f" Augmentacje:     erasing={erasing}, hsv_s={hsv_s}, hsv_v={hsv_v}")
        print(f" Learning Rate:   lr0={lr0}, lrf={lrf}")
        print(f" Katalog wyjścia: {self.stage4_dir}")
        print("=" * 78)

        self.stage4_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(stage3_weight), task="detect")

        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            erasing=erasing,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            device=self.device,
            workers=self.workers,
            project=str(self.stage4_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage4_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage4_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage4_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku wag best.pt w: {self.stage4_dir}")

        best_pt = self.stage4_dir / "best.pt"
        best_pth = self.stage4_dir / "best.pth"
        last_pt = self.stage4_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 4] Zapisano checkpointy: {best_pt} oraz {last_pt}")

        # Ewaluacja
        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage4"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 4 (AUGMENTATION)", metrics)
        return best_pt

    def run_stage5(
        self,
        stage4_weight: Path,
        epochs: int = 8,
        lr0: float = 0.0003,
        lrf: float = 0.1,
        optimizer: str = "AdamW",
    ) -> Path:
        """
        Stage 5: Trening na zdjęciach bez koloru żółtego (desaturacja kwiatów, zachowanie liści).
        """
        from ultralytics import YOLO

        print("\n" + "=" * 78)
        print(" STAGE 5 — TRENING NA ZDJĘCIACH BEZ KOLORU ŻÓŁTEGO (NO-YELLOW)")
        print(f" Model wejściowy: {stage4_weight}")
        print(f" Epoki:           {epochs}")
        print(f" Learning Rate:   lr0={lr0}, lrf={lrf}")
        print(f" Katalog wyjścia: {self.stage5_dir}")
        print("=" * 78)

        self.stage5_dir.mkdir(parents=True, exist_ok=True)

        # 1. Przygotowanie odrębnego datasetu No-Yellow
        source_train_manifest = DATASETS_DIR / "stage2_train.txt"
        dataset_out = self.stage5_dir / "dataset"
        no_yellow_yaml = prepare_no_yellow_dataset(
            source_manifest=source_train_manifest,
            output_dataset_dir=dataset_out,
            val_manifest=self.val_manifest,
        )

        # 2. Trening modelu
        model = YOLO(str(stage4_weight), task="detect")

        model.train(
            data=str(no_yellow_yaml),
            epochs=epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            device=self.device,
            workers=self.workers,
            project=str(self.stage5_dir),
            name="run",
            exist_ok=True,
            save=True,
            verbose=True,
        )

        best_trained = self.stage5_dir / "run" / "weights" / "best.pt"
        last_trained = self.stage5_dir / "run" / "weights" / "last.pt"
        if not best_trained.is_file():
            cands = list(self.stage5_dir.rglob("best.pt"))
            best_trained = cands[0] if cands else None

        if not best_trained or not best_trained.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku wag best.pt w: {self.stage5_dir}")

        best_pt = self.stage5_dir / "best.pt"
        best_pth = self.stage5_dir / "best.pth"
        last_pt = self.stage5_dir / "last.pt"
        shutil.copy2(best_trained, best_pt)
        if last_trained.is_file():
            shutil.copy2(last_trained, last_pt)

        try:
            if best_pth.exists() or best_pth.is_symlink():
                best_pth.unlink()
            best_pth.symlink_to(best_pt.name)
        except Exception:
            shutil.copy2(best_trained, best_pth)

        print(f"\n[Stage 5] Zapisano finalny model: {best_pt} oraz {last_pt}")

        # Ewaluacja na standardowym zbiorze walidacyjnym (weryfikacja czy model rozpoznaje starca w naturze)
        metrics = evaluate_checkpoint(
            model_weight=best_pt,
            data_yaml=self.data_yaml,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
        )
        self.metrics_summary["stage5"] = {
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }
        self._print_metrics_block("STAGE 5 (NO-YELLOW FLOWERS)", metrics)
        return best_pt

    def save_comparison_summary(self):
        """Zapisuje raport końcowy Stage 2 -> Stage 3 -> Stage 4 -> Stage 5."""
        summary_dir = self.output_dir / "leaves_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        json_path = summary_dir / "metrics_comparison.json"
        csv_path = summary_dir / "metrics_comparison.csv"
        txt_path = summary_dir / "summary.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)

        csv_lines = ["stage,checkpoint,mAP50,mAP50-95,precision,recall,fitness"]
        stages = ["stage2", "stage3", "stage4", "stage5"]
        for st in stages:
            if st in self.metrics_summary:
                info = self.metrics_summary[st]
                m = info["metrics"]
                csv_lines.append(
                    f"{st},{info['checkpoint']},{m.get('mAP50', 0):.4f},"
                    f"{m.get('mAP50-95', 0):.4f},{m.get('precision', 0):.4f},"
                    f"{m.get('recall', 0):.4f},{m.get('fitness', 0):.4f}"
                )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        # Tabela tekstowa
        header = "=" * 86 + "\n"
        title = "       PODSUMOWANIE ETAPÓW LIŚCI: STAGE 2 -> STAGE 3 -> STAGE 4 -> STAGE 5\n"
        sep = "-" * 86 + "\n"
        col_names = f"{'Etap':<18} | {'mAP@50':<10} | {'mAP@50-95':<12} | {'Precyzja':<10} | {'Czułość':<10} | {'Fitness':<10}\n"

        labels_map = {
            "stage2": "Stage 2 (Base)",
            "stage3": "Stage 3 (1024)",
            "stage4": "Stage 4 (Aug)",
            "stage5": "Stage 5 (NoYel)",
        }

        rows = []
        for st in stages:
            if st in self.metrics_summary:
                m = self.metrics_summary[st]["metrics"]
                lbl = labels_map.get(st, st)
                row = (
                    f"{lbl:<18} | "
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
        print(f"[Podsumowanie] Zapisano w:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
        print(f"  - {txt_path}")

    def _print_metrics_block(self, title: str, metrics: Dict[str, float]):
        print(f"\n--- WYNIKI WALIDACJI: {title} ---")
        print(f"  mAP@50               : {metrics.get('mAP50', 0.0):.4f}")
        print(f"  mAP@50-95            : {metrics.get('mAP50-95', 0.0):.4f}")
        print(f"  Precyzja (Precision) : {metrics.get('precision', 0.0):.4f}")
        print(f"  Czułość (Recall)     : {metrics.get('recall', 0.0):.4f}")
        print(f"  Fitness              : {metrics.get('fitness', 0.0):.4f}")
        print("---------------------------------------")


def run_leaves_fine_tuning(
    stage2_model_path: Optional[Union[str, Path]] = None,
    data_yaml: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    stage3_epochs: int = 8,
    stage4_epochs: int = 8,
    stage5_epochs: int = 8,
    lr0_stage3: float = 0.0005,
    lr0_stage4: float = 0.0003,
    lr0_stage5: float = 0.0003,
    imgsz: int = 1024,
    batch: int = 8,
    device: Optional[str] = None,
    workers: int = 4,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
    stage3_checkpoint: Optional[Union[str, Path]] = None,
    stage4_checkpoint: Optional[Union[str, Path]] = None,
    val_only: bool = False,
):
    """Główna funkcja wykonawcza potoku Stage 3 -> Stage 4 -> Stage 5."""
    resolved_model = resolve_model_path(stage2_model_path or STAGE2_DEFAULT_MODEL)
    inspect_model_architecture(resolved_model)

    pipeline = LeavesFineTuningPipeline(
        stage2_model_path=resolved_model,
        data_yaml=data_yaml,
        output_dir=output_dir,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
    )

    # 0. Ewaluacja wejściowego modelu Stage 2
    pipeline.evaluate_initial_stage2()
    if val_only:
        print("[Info] Uruchomiono w trybie --val-only. Zakończono.")
        return pipeline.metrics_summary

    # 1. Stage 3
    if skip_stage3:
        best_s3 = Path(stage3_checkpoint or (pipeline.stage3_dir / "best.pt")).resolve()
        print(f"\n[Info] Pomijam trening Stage 3. Używam istniejącego: {best_s3}")
        m = evaluate_checkpoint(best_s3, pipeline.data_yaml, device=pipeline.device, imgsz=imgsz, batch=batch)
        pipeline.metrics_summary["stage3"] = {"checkpoint": str(best_s3), "metrics": m}
    else:
        best_s3 = pipeline.run_stage3(
            epochs=stage3_epochs,
            lr0=lr0_stage3,
        )

    # 2. Stage 4
    if skip_stage4:
        best_s4 = Path(stage4_checkpoint or (pipeline.stage4_dir / "best.pt")).resolve()
        print(f"\n[Info] Pomijam trening Stage 4. Używam istniejącego: {best_s4}")
        m = evaluate_checkpoint(best_s4, pipeline.data_yaml, device=pipeline.device, imgsz=imgsz, batch=batch)
        pipeline.metrics_summary["stage4"] = {"checkpoint": str(best_s4), "metrics": m}
    else:
        best_s4 = pipeline.run_stage4(
            stage3_weight=best_s3,
            epochs=stage4_epochs,
            lr0=lr0_stage4,
        )

    # 3. Stage 5
    pipeline.run_stage5(
        stage4_weight=best_s4,
        epochs=stage5_epochs,
        lr0=lr0_stage5,
    )

    # 4. Podsumowanie
    pipeline.save_comparison_summary()
    return pipeline.metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning modeli Ragwort ukierunkowany na cechy liści (Stage 3 -> 4 -> 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(STAGE2_DEFAULT_MODEL),
        help="Ścieżka do najlepszego checkpointu z etapu 2",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DATASETS_DIR / "stage2_data.yaml",
        help="Plik konfiguracji danych z poprzedniego etapu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINE_RESULTS_DIR,
        help="Główny katalog zapisu wyników",
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=8,
        help="Liczba epok dla etapu 3 (rozdzielczość 1024)",
    )
    parser.add_argument(
        "--stage4-epochs",
        type=int,
        default=8,
        help="Liczba epok dla etapu 4 (augmentacje liści)",
    )
    parser.add_argument(
        "--stage5-epochs",
        type=int,
        default=8,
        help="Liczba epok dla etapu 5 (dane bez koloru żółtego)",
    )
    parser.add_argument(
        "--lr0-stage3",
        type=float,
        default=0.0005,
        help="Learning rate dla etapu 3",
    )
    parser.add_argument(
        "--lr0-stage4",
        type=float,
        default=0.0003,
        help="Learning rate dla etapu 4",
    )
    parser.add_argument(
        "--lr0-stage5",
        type=float,
        default=0.0003,
        help="Learning rate dla etapu 5",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Rozdzielczość obrazu dla etapów 3, 4 i 5",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Rozmiar batcha (bezpieczny dla 6GB VRAM przy 1024)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Urządzenie obliczeniowe ('cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Liczba procesów roboczych DataLoader",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Pomiń etap 3 i przejdź do etapu 4",
    )
    parser.add_argument(
        "--skip-stage4",
        action="store_true",
        help="Pomiń etap 4 i przejdź do etapu 5",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        type=str,
        default=None,
        help="Ścieżka do checkpointu etapu 3 w przypadku pominięcia",
    )
    parser.add_argument(
        "--stage4-checkpoint",
        type=str,
        default=None,
        help="Ścieżka do checkpointu etapu 4 w przypadku pominięcia",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Tylko ewaluacja bez uruchamiania treningu",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_leaves_fine_tuning(
        stage2_model_path=args.model,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        stage3_epochs=args.stage3_epochs,
        stage4_epochs=args.stage4_epochs,
        stage5_epochs=args.stage5_epochs,
        lr0_stage3=args.lr0_stage3,
        lr0_stage4=args.lr0_stage4,
        lr0_stage5=args.lr0_stage5,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        skip_stage3=args.skip_stage3,
        skip_stage4=args.skip_stage4,
        stage3_checkpoint=args.stage3_checkpoint,
        stage4_checkpoint=args.stage4_checkpoint,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()
