"""
app.py - Serwer aplikacji webowej dla projektu Ragwort Detection.
Obsluguje:
  1. Podglad wideo na zywo z kamery (lub symulacji) z nalozonym pipeline'em Opcji C
     (segmentacja organow: kwiaty, liscie, lodygi + orzeczenie czy roslina to starzec jakubek).
  2. Przetwarzanie pojedynczych zdjec (upload z dysku i analiza biometryczna).
  3. Panel treningowy (generator komend na inny komputer GPU / Colab oraz monitoring).
  4. Dynamiczne przelaczanie wag modeli w locie (segmentacja vs detekcja).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

# Ustalenie sciezek projektu
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_evaluation.train_ragwort_segmentation import RagwortPlantRecognizer

# Katalogi modeli i wag
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # Maks 64MB dla uploadu zdjec/wag


class VisionPipeline:
    """Zarzadza modelem wizyjnym, kamera oraz pipeline'em Opcji C."""

    def __init__(self):
        self.conf_threshold = 0.30
        self.iou_threshold = 0.45
        self.camera_index = 0
        self.use_simulation = False
        self.fps = 0.0
        self.only_ragwort = False
        self.model_path: Path = self._find_initial_weights()
        self.model: Optional[YOLO] = None
        self.is_segmentation = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.recognizer = RagwortPlantRecognizer()

        # Telemetria biezaca
        self.latest_analysis: Dict[str, Any] = {
            "is_ragwort": False,
            "confidence": 0.0,
            "stage": "unknown",
            "counts": {"flowers": 0, "leaves": 0, "stems": 0, "total_parts": 0},
            "areas": {"flower_area_px": 0.0, "leaf_area_px": 0.0, "stem_area_px": 0.0},
        }
        self.latest_detections: List[Dict[str, Any]] = []

        self.load_model(self.model_path)
        self.init_camera()

    def _find_initial_weights(self) -> Path:
        """Wybiera najlepsze dostepne wagi w projekcie."""
        candidates = [
            MODELS_DIR / "ragwort_segmentation_best.pt",
            PROJECT_ROOT / "yolov8n-seg.pt",
            MODELS_DIR / "ragwort_yolov8_best.pt",
            WEIGHTS_DIR / "ragwort_yolov8_best.pt",
            PROJECT_ROOT / "yolov8n.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return PROJECT_ROOT / "yolov8n-seg.pt"

    def load_model(self, path_or_name: Union[str, Path]) -> bool:
        """Laduje lub przelacza model YOLO."""
        try:
            target = Path(path_or_name)
            if not target.is_absolute() and not str(path_or_name).endswith(".pt"):
                # Np. yolov8n-seg.pt
                target_str = str(path_or_name)
            else:
                target_str = str(target)

            print(f"[VisionPipeline] Ladowanie wag: {target_str}")
            self.model = YOLO(target_str)
            self.model_path = Path(target_str)

            # Sprawdzenie czy model obsluguje zadanie segmentacji
            task_type = getattr(self.model, "task", "detect")
            self.is_segmentation = ("seg" in str(target_str).lower()) or (task_type == "segment")
            print(f"[VisionPipeline] Model zaladowany. Typ: {'Segmentacja (Opcja C)' if self.is_segmentation else 'Detekcja (BBoxes)'}")
            return True
        except Exception as e:
            print(f"[VisionPipeline] Blad ladowania modelu: {e}")
            return False

    def init_camera(self, index: int = 0, force_simulation: bool = False):
        """Inicjalizuje kamere fizyczna lub tryb symulacji."""
        self.camera_index = index
        self.use_simulation = force_simulation

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if not self.use_simulation:
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    print(f"[Kamera] Brak kamery pod indeksem {self.camera_index}. Przelaczam na symulacje.")
                    self.use_simulation = True
                    self.cap = None
                else:
                    print(f"[Kamera] Kamera fizyczna #{self.camera_index} otwarta pomyslnie.")
            except Exception as e:
                print(f"[Kamera] Wyjatek przy otwarciu kamery: {e}")
                self.use_simulation = True
                self.cap = None

    def get_raw_frame(self) -> np.ndarray:
        """Pobiera surowa klatke z kamery lub testowego zdjecia laki."""
        if not self.use_simulation and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        # Tryb symulacyjny: ladowanie zdjecia testowego laki ze starcem
        test_images = [
            PROJECT_ROOT / "data" / "external_test_images" / "ragwort_field_1.jpg",
            PROJECT_ROOT / "data" / "external_test_images" / "ragwort_flowers_2.jpg",
        ]
        for t in test_images:
            if t.exists():
                f = cv2.imread(str(t))
                if f is not None:
                    return cv2.resize(f, (640, 480))

        # Ostateczny fallback: generowana klatka z napisem
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy[:] = (30, 60, 30)  # Ciemna zielen laki
        cv2.putText(dummy, "Symulacja Kamery: Brak urzadzenia USB", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return dummy

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Przetwarza pojedyncza klatke przez model i modul decyzyjny Opcji C.
        Zwraca: (obraz z nalozonymi maskami i HUD, wynik analizy organow, lista detekcji dla robota).
        """
        if self.model is None:
            return frame, self.latest_analysis, []

        h, w = frame.shape[:2]
        res = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        labels = res.boxes.cls.cpu().numpy().astype(int).tolist() if res.boxes is not None else []
        names_dict = res.names or {}
        class_names = [names_dict.get(cid, str(cid)) for cid in labels]

        masks_xy: List[np.ndarray] = []
        if res.masks is not None and hasattr(res.masks, "xy"):
            masks_xy = [p for p in res.masks.xy]

        prediction_dict = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": class_names,
            "masks": masks_xy,
            "orig_shape": (h, w),
        }

        # 1. Analiza Opcji C (biometria organow i decyzja o starcu)
        analysis = self.recognizer.analyze_organs(prediction_dict)

        # 2. Renderowanie masek i wizualizacji
        overlay = frame.copy()
        detections: List[Dict[str, Any]] = []

        for i, (box, score, cname) in enumerate(zip(boxes, scores, class_names)):
            name_lower = str(cname).lower()
            if "flower" in name_lower or "kwiat" in name_lower:
                color = self.recognizer.ORGAN_COLORS["flower"]
            elif "leaf" in name_lower or "lisc" in name_lower:
                color = self.recognizer.ORGAN_COLORS["leaf"]
            elif "stem" in name_lower or "lodyg" in name_lower:
                color = self.recognizer.ORGAN_COLORS["stem"]
            elif "ragwort" in name_lower:
                color = self.recognizer.ORGAN_COLORS["ragwort"]
            else:
                color = self.recognizer.ORGAN_COLORS["default"]

            # Rysowanie maski poligonowej
            if i < len(masks_xy) and len(masks_xy[i]) >= 3:
                pts = masks_xy[i].astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Rysowanie ramki i etykiety
            if len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Punkt celowniczy dla manipulatora robota Hege
                cv2.circle(frame, (xc, yc), 4, (0, 255, 255), -1)
                cv2.drawMarker(frame, (xc, yc), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

                label_txt = f"{cname} {score*100:.0f}%"
                cv2.putText(frame, label_txt, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                detections.append({
                    "class": cname,
                    "confidence": round(float(score), 2),
                    "box": [x1, y1, x2, y2],
                    "center": [xc, yc],
                })

        # Mieszanie masek poligonowych (alpha blending)
        if len(masks_xy) > 0:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 3. Pasek naglowkowy HUD (Wskaznik Opcji C)
        hud_height = 42
        hud = np.zeros((hud_height, w, 3), dtype=np.uint8)

        if analysis["is_ragwort"]:
            hud[:] = (15, 110, 25)  # Ciemna zielen
            verdict_str = f"WERDYKT: STARZEC JAKUBEK ({analysis['confidence']*100:.0f}%) | {analysis['stage'].upper()}"
        else:
            hud[:] = (20, 20, 110)  # Ciemna czerwien
            verdict_str = "WERDYKT: BRAK STARCA / INNA ROSLINNOSC"

        counts = analysis["counts"]
        details_str = f"Organy: {counts['flowers']} kwiaty | {counts['leaves']} liscie | {counts['stems']} lodygi"

        cv2.putText(hud, verdict_str, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, details_str, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        combined = np.vstack([hud, frame])
        return combined, analysis, detections


# Inicjalizacja globalnego pipeline'u
pipeline = VisionPipeline()


def generate_video_stream():
    """Generator strumienia MJPEG dla widoku kamery w przegladarce."""
    prev_time = time.time()

    while True:
        raw_frame = pipeline.get_raw_frame()

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time
        pipeline.fps = round(fps, 1)

        processed_frame, analysis, detections = pipeline.process_frame(raw_frame)
        pipeline.latest_analysis = analysis
        pipeline.latest_detections = detections

        # Znak wodny FPS w dolnym rogu
        h, w = processed_frame.shape[:2]
        cv2.putText(processed_frame, f"FPS: {pipeline.fps}", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# --------------------------------------------------------------------------
# Trasy serwera Flask
# --------------------------------------------------------------------------

@app.route("/")
def index():
    """Glowny widok aplikacji webowej."""
    return render_template(
        "index.html",
        model_name=pipeline.model_path.name,
        is_segmentation=pipeline.is_segmentation,
        conf_default=pipeline.conf_threshold,
        iou_default=pipeline.iou_threshold,
    )


@app.route("/video_feed")
def video_feed():
    """Strumien MJPEG bezposrednio dla tagu <img>."""
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    """Zwraca pelne dane telemetryczne i wyniki analizy Opcji C dla robota Hege."""
    return jsonify({
        "fps": pipeline.fps,
        "model": pipeline.model_path.name,
        "is_segmentation": pipeline.is_segmentation,
        "camera_mode": "simulation" if pipeline.use_simulation else f"camera_{pipeline.camera_index}",
        "analysis": pipeline.latest_analysis,
        "detections": pipeline.latest_detections,
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    """Zmiana progow detekcji oraz zrodla kamery w locie."""
    data = request.json or {}
    if "conf" in data:
        pipeline.conf_threshold = float(data["conf"])
    if "iou" in data:
        pipeline.iou_threshold = float(data["iou"])
    if "camera_mode" in data:
        mode = str(data["camera_mode"])
        if mode == "simulation":
            pipeline.init_camera(force_simulation=True)
        elif mode.isdigit():
            pipeline.init_camera(index=int(mode), force_simulation=False)

    return jsonify({
        "status": "success",
        "conf": pipeline.conf_threshold,
        "iou": pipeline.iou_threshold,
        "use_simulation": pipeline.use_simulation,
    })


@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Lista dostepnych modeli w projekcie."""
    available_files = []

    # Szukanie w models/ i weights/ i katalogu glownym
    for folder in [MODELS_DIR, WEIGHTS_DIR, PROJECT_ROOT]:
        if folder.exists():
            for f in folder.glob("*.pt"):
                available_files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                    "type": "segment" if "seg" in f.name.lower() else "detect"
                })

    return jsonify({
        "active_model": pipeline.model_path.name,
        "is_segmentation": pipeline.is_segmentation,
        "models": available_files
    })


@app.route("/api/models/switch", methods=["POST"])
def api_switch_model():
    """Przelacza aktywny model w pipeline."""
    data = request.json or {}
    target = data.get("model")
    if not target:
        return jsonify({"status": "error", "message": "Brak parametru model"}), 400

    # Szukamy po sciezce lub nazwie
    target_path = Path(target)
    if not target_path.exists():
        for folder in [MODELS_DIR, WEIGHTS_DIR, PROJECT_ROOT]:
            candidate = folder / target
            if candidate.exists():
                target_path = candidate
                break

    success = pipeline.load_model(target_path)
    return jsonify({
        "status": "success" if success else "error",
        "active_model": pipeline.model_path.name,
        "is_segmentation": pipeline.is_segmentation,
    })


@app.route("/api/process_image", methods=["POST"])
def api_process_image():
    """
    Endpoint do przetwarzania pojedynczego zdjecia wgranego przez uzytkownika.
    Wykonuje pelny pipeline Opcji C i zwraca przetworzony obraz Base64 oraz statystyki.
    """
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Brak pliku zdjecia"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"status": "error", "message": "Nie udalo sie zdekodowac obrazu"}), 400

    processed_img, analysis, detections = pipeline.process_frame(image)

    # Konwersja wyniku do JPEG base64 dla przegladarki
    _, buffer = cv2.imencode(".jpg", processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "status": "success",
        "analysis": analysis,
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
    })


@app.route("/api/training/command", methods=["POST"])
def api_training_command():
    """
    Generuje gotowa komende treningowa do uruchomienia na innym komputerze z GPU / Colab.
    """
    data = request.json or {}
    model_name = data.get("model", "yolov8n-seg.pt")
    epochs = int(data.get("epochs", 50))
    batch = int(data.get("batch", 8))
    imgsz = int(data.get("imgsz", 640))
    device = data.get("device", "0")

    cmd_shell = (
        f"python scripts/train_segmentation.py "
        f"--model {model_name} "
        f"--data data/ragwort_parts.yaml "
        f"--epochs {epochs} "
        f"--batch {batch} "
        f"--imgsz {imgsz} "
        f"--device {device}"
    )

    colab_script = f"""# --- Uruchomienie na Google Colab (GPU) ---
!git clone <URL_TWOJEGO_REPOZYTORIUM>
%cd RagwortDetection
!pip install -r requirements/requirementsYOLO.txt
!python scripts/train_segmentation.py --model {model_name} --data data/ragwort_parts.yaml --epochs {epochs} --batch {batch} --imgsz {imgsz} --device 0
"""

    return jsonify({
        "command_shell": cmd_shell,
        "colab_script": colab_script,
        "output_weight_expected": "models/ragwort_segmentation_best.pt"
    })


# --------------------------------------------------------------------------
# Start serwera
# --------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"===========================================================")
    print(f"  🌿 Ragwort Detection & Option C Vision Dashboard")
    print(f"  Adres: http://localhost:{port}")
    print(f"  Aktywny model: {pipeline.model_path.name}")
    print(f"  Typ: {'Segmentacja Instancji' if pipeline.is_segmentation else 'Detekcja'}")
    print(f"===========================================================")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
