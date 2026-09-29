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
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    rs = None
    HAVE_REALSENSE = False


def get_connected_realsense_device() -> Optional[str]:
    """Wykrywa podlaczone urzadzenie Intel RealSense (np. D405)."""
    if not HAVE_REALSENSE or rs is None:
        return None
    try:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            dev = devs[0]
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            return f"{name} (S/N: {sn})"
    except Exception as e:
        print(f"[RealSense Check] Blad sprawdzania: {e}")
    return None

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

    RESOLUTIONS = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "4k": (3840, 2160),
        "640p": (640, 480),
    }

    def __init__(self):
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.camera_index = 0

        # Sprawdzenie obecnosci RealSense D405
        rs_dev = get_connected_realsense_device()
        self.realsense_device_name = rs_dev
        self.camera_mode = "realsense" if rs_dev else "auto"
        self.camera_resolution = "720p" if rs_dev else "1080p"
        self.frame_width = 1280 if rs_dev else 1920
        self.frame_height = 720 if rs_dev else 1080
        self.inference_imgsz = 640  # Model wnioskuje w 640 lub 1080
        self.use_simulation = False
        self.realsense_pipeline = None
        self.realsense_align = None
        self.latest_depth_array: Optional[np.ndarray] = None
        self.fps = 0.0
        self.only_ragwort = False

        # Parametry kamery (Hardware ISP RealSense D405 + Cyfrowa normalizacja CLAHE)
        self.camera_settings: Dict[str, Any] = {
            "auto_exposure": True,
            "exposure": 33000,
            "gain": 16,
            "auto_white_balance": True,
            "white_balance": 4600,
            "contrast": 50,
            "brightness": 0,
            "saturation": 64,
            "gamma": 300,
            "sharpness": 50,
            "backlight_compensation": False,
            "clahe_enabled": False,
            "clahe_clip": 2.0,
        }

        self.model_family: str = "yolo"
        self.model_type: str = "YOLOv8 Segmentation (Option C)"
        self.model_path: Path = self._find_initial_weights()
        self.model: Optional[Any] = None
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
        self.init_camera(mode=self.camera_mode)

    def _find_initial_weights(self) -> Path:
        """Wybiera najlepsze dostepne wagi w projekcie (priorytet: Segmentacja Opcji C)."""
        candidates = [
            MODELS_DIR / "ragwort_segmentation_best.pt",
            MODELS_DIR / "ragwort_yolov8_seg_best.pt",
            MODELS_DIR / "ragwort_yolov8s_best.pt",
            WEIGHTS_DIR / "checkpoint0044.pth",
            WEIGHTS_DIR / "model.pt",
            MODELS_DIR / "deim_dinov3_ragwort.pt",
            WEIGHTS_DIR / "best (1).pt",
            WEIGHTS_DIR / "best.pt",
            WEIGHTS_DIR / "ragwort_yolov8_best (1).pt",
            WEIGHTS_DIR / "ragwort_yolov8_best.pt",
            MODELS_DIR / "ragwort_yolov8_best.pt",
            PROJECT_ROOT / "yolov8s-seg.pt",
            PROJECT_ROOT / "yolov8s.pt",
            PROJECT_ROOT / "yolov8n-seg.pt",
            PROJECT_ROOT / "yolov8n.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return PROJECT_ROOT / "yolov8s-seg.pt"

    def load_model(self, path_or_name: Union[str, Path], force_family: Optional[str] = None) -> bool:
        """Laduje lub przelacza model YOLO lub DEIMv2 (.pt oraz .pth)."""
        try:
            target = Path(path_or_name)
            if not target.exists():
                for folder in [MODELS_DIR, WEIGHTS_DIR, APP_DIR / "weights", PROJECT_ROOT]:
                    candidate = folder / path_or_name
                    if candidate.exists():
                        target = candidate
                        break
                    candidate_name = folder / Path(path_or_name).name
                    if candidate_name.exists():
                        target = candidate_name
                        break

            target_str = str(target)
            print(f"[VisionPipeline] Ladowanie wag: {target_str} (Zadana rodzina: {force_family})")

            # Rozpoznanie czy model nalezy do rodziny DEIM czy YOLO
            is_deim = False
            if force_family == "deim":
                is_deim = True
            elif force_family == "yolo":
                is_deim = False
            else:
                name_l = target.name.lower()
                if "deim" in name_l or target_str.endswith("model.pt") or name_l == "model.pt" or name_l.endswith(".pth"):
                    is_deim = True
                elif Path(target_str).exists():
                    try:
                        chk = torch.load(target_str, map_location="cpu", weights_only=False)
                        if isinstance(chk, dict) and ("DINOv3STAs" in str(chk.get("config", "")) or chk.get("format") == "DEIMv2-DINOv3" or "checkpoint" in str(name_l)):
                            is_deim = True
                    except Exception:
                        pass

            if is_deim:
                from src.models.deim import DEIMModel
                self.model = DEIMModel(target_str)
                self.model_family = "deim"
                is_pth = target.name.lower().endswith(".pth")
                self.model_type = "DEIMv2 + DINOv3 (PyTorch .pth)" if is_pth else "DEIMv2 + DINOv3 Transformer"
                self.is_segmentation = False
            else:
                self.model = YOLO(target_str)
                self.model_family = "yolo"
                task_type = getattr(self.model, "task", "detect")
                is_seg = ("seg" in str(target_str).lower()) or (task_type == "segment")
                self.is_segmentation = is_seg
                name_lower = Path(target_str).name.lower()
                if is_seg:
                    self.model_type = "YOLOv8 Segmentation (Option C)"
                elif "8s" in name_lower or "best" in name_lower or (Path(target_str).exists() and 18 * 1024 * 1024 < Path(target_str).stat().st_size < 100 * 1024 * 1024):
                    self.model_type = "YOLOv8s Weed Detection"
                elif "8n" in name_lower:
                    self.model_type = "YOLOv8n Weed Detection"
                else:
                    self.model_type = "YOLOv8 Detection"

            self.model_path = Path(target_str)
            print(f"[VisionPipeline] Model zaladowany: {self.model_path.name} | Rodzina: {self.model_family} | Typ: {self.model_type} | Segmentacja: {self.is_segmentation}")
            return True
        except Exception as e:
            print(f"[VisionPipeline] Blad ladowania modelu: {e}")
            return False

    def _release_all_cameras(self):
        """Zwalnia potok RealSense oraz kamery OpenCV."""
        if getattr(self, "realsense_pipeline", None) is not None:
            try:
                self.realsense_pipeline.stop()
            except Exception:
                pass
            self.realsense_pipeline = None
            self.realsense_align = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def set_camera_resolution(self, resolution: str):
        """Ustawia docelowa rozdzielczosc strumienia i synchronizuje wymiary ramki."""
        if resolution in self.RESOLUTIONS:
            self.camera_resolution = resolution
            target_w, target_h = self.RESOLUTIONS[resolution]
            self.frame_width = target_w
            self.frame_height = target_h
            if self.cap is not None and self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                except Exception:
                    pass
            print(f"[Camera] Set target resolution to: {resolution} ({target_w}x{target_h})")

    def init_camera(self, mode: str = "auto", force_simulation: bool = False, resolution: str = "1080p"):
        """Inicjalizuje kamere: RealSense D405 (RGB-D), kamere USB lub tryb symulacji."""
        self._release_all_cameras()

        if resolution in self.RESOLUTIONS:
            self.camera_resolution = resolution

        target_w, target_h = self.RESOLUTIONS.get(self.camera_resolution, (1920, 1080))
        self.frame_width = target_w
        self.frame_height = target_h

        if force_simulation or mode == "simulation":
            self.camera_mode = "simulation"
            self.use_simulation = True
            self.frame_width = target_w
            self.frame_height = target_h
            print("[Kamera] Uruchomiono tryb symulacji.")
            return

        # Automatyczne wykrywanie: jesli RealSense D405 jest podlaczony, bierzemy go priorytetowo
        if mode == "auto":
            rs_dev = get_connected_realsense_device()
            mode = "realsense" if rs_dev else "0"

        # 1. Obsluga kamery Intel RealSense D405 (RGB-D)
        if mode == "realsense" and HAVE_REALSENSE:
            try:
                rs_dev = get_connected_realsense_device()
                if rs_dev:
                    print(f"[RealSense D405] Otwieranie kamery {rs_dev}...")
                    p = rs.pipeline()
                    cfg = rs.config()
                    # D405 natywne profile RGB-D: 1280x720 @ 30fps
                    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                    self.realsense_align = rs.align(rs.stream.color)
                    p.start(cfg)
                    self.realsense_pipeline = p
                    self.realsense_device_name = rs_dev
                    self.camera_mode = "realsense"
                    self.use_simulation = False
                    self.frame_width = target_w
                    self.frame_height = target_h
                    print(f"[RealSense D405] Success: RGB-D active ({target_w}x{target_h} / native 1280x720 @ 30fps)")
                    # Apply current exposure, white balance and contrast settings
                    self.apply_camera_settings(self.camera_settings)
                    return
                else:
                    print("[RealSense] Brak podlaczonej kamery D405. Przelaczam na USB / symulacje.")
                    mode = "0"
            except Exception as e:
                print(f"[RealSense D405] Blad otwarcia potoku RealSense: {e}. Proba kamery USB...")
                self.realsense_pipeline = None
                mode = "0"

        # 2. Obsluga tradycyjnej kamery USB (OpenCV)
        clean_idx = mode.replace("camera_", "")
        cam_idx = int(clean_idx) if clean_idx.isdigit() else 0
        self.camera_index = cam_idx
        self.camera_mode = f"camera_{cam_idx}"

        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"[Kamera] Brak kamery pod indeksem {self.camera_index}. Przelaczam na symulacje.")
                self.camera_mode = "simulation"
                self.use_simulation = True
                self.cap = None
                self.frame_width = target_w
                self.frame_height = target_h
            else:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_width = actual_w if actual_w > 0 else target_w
                self.frame_height = actual_h if actual_h > 0 else target_h
                self.use_simulation = False
                print(f"[Kamera] Kamera USB #{self.camera_index} otwarta. Rozdzielczosc: {self.frame_width}x{self.frame_height}")
        except Exception as e:
            print(f"[Kamera] Wyjatek przy otwarciu kamery: {e}")
            self.camera_mode = "simulation"
            self.use_simulation = True
            self.cap = None
            self.frame_width = target_w
            self.frame_height = target_h

    def get_realsense_sensor(self):
        """Zwraca aktywny sensor RealSense kontrolujacy rejestry ISP."""
        if not HAVE_REALSENSE or self.realsense_pipeline is None:
            return None
        try:
            dev = self.realsense_pipeline.get_active_profile().get_device()
            if len(dev.sensors) > 0:
                return dev.sensors[0]
        except Exception:
            pass
        return None

    def apply_camera_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Aplikuje parametry sprzetowe ISP do RealSense D405 oraz ustawienia cyfrowe."""
        sensor = self.get_realsense_sensor()

        for k, v in new_settings.items():
            if k in self.camera_settings:
                self.camera_settings[k] = v

        if sensor is not None and HAVE_REALSENSE and rs is not None:
            try:
                # 1. Auto-Exposure vs Manual Exposure
                if "auto_exposure" in new_settings and sensor.supports(rs.option.enable_auto_exposure):
                    ae_val = 1.0 if new_settings["auto_exposure"] else 0.0
                    sensor.set_option(rs.option.enable_auto_exposure, ae_val)

                if not self.camera_settings.get("auto_exposure", True):
                    if "exposure" in new_settings and sensor.supports(rs.option.exposure):
                        val = max(1.0, min(165000.0, float(new_settings["exposure"])))
                        sensor.set_option(rs.option.exposure, val)
                    if "gain" in new_settings and sensor.supports(rs.option.gain):
                        val = max(16.0, min(248.0, float(new_settings["gain"])))
                        sensor.set_option(rs.option.gain, val)

                # 2. Auto-White-Balance vs Manual White Balance
                if "auto_white_balance" in new_settings and sensor.supports(rs.option.enable_auto_white_balance):
                    awb_val = 1.0 if new_settings["auto_white_balance"] else 0.0
                    sensor.set_option(rs.option.enable_auto_white_balance, awb_val)

                if not self.camera_settings.get("auto_white_balance", True):
                    if "white_balance" in new_settings and sensor.supports(rs.option.white_balance):
                        val = max(2800.0, min(6500.0, float(new_settings["white_balance"])))
                        sensor.set_option(rs.option.white_balance, val)

                # 3. Parametry jakosci obrazu (Kontrast, Jasnosc, Nasycenie, Gamma, Ostrosc)
                hw_params = [
                    ("contrast", rs.option.contrast, 0.0, 100.0),
                    ("brightness", rs.option.brightness, -64.0, 64.0),
                    ("saturation", rs.option.saturation, 0.0, 100.0),
                    ("gamma", rs.option.gamma, 100.0, 500.0),
                    ("sharpness", rs.option.sharpness, 0.0, 100.0),
                ]
                for key, opt, min_v, max_v in hw_params:
                    if key in new_settings and sensor.supports(opt):
                        val = max(min_v, min(max_v, float(new_settings[key])))
                        sensor.set_option(opt, val)

                # 4. Kompensacja tylnego oswietlenia
                if "backlight_compensation" in new_settings and sensor.supports(rs.option.backlight_compensation):
                    bc_val = 1.0 if new_settings["backlight_compensation"] else 0.0
                    sensor.set_option(rs.option.backlight_compensation, bc_val)

            except Exception as e:
                print(f"[RealSense ISP] Blad zapisu rejestru: {e}")

        return self.camera_settings

    def normalize_illumination(self, img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Stabilizuje oswietlenie (slonce/cien) przez adaptacyjna korekcje luminancji LAB (CLAHE)."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception as e:
            return img

    def get_raw_frame(self) -> np.ndarray:
        """Pobiera surowa klatke z RealSense D405, kamery USB lub symulacji ze skalowaniem do zadanej rozdzielczosci."""
        frame = None
        target_w, target_h = self.RESOLUTIONS.get(self.camera_resolution, (1280, 720))

        # 1. Strumien RealSense D405 (RGB-D)
        if not self.use_simulation and self.camera_mode == "realsense" and self.realsense_pipeline is not None:
            try:
                frames = self.realsense_pipeline.wait_for_frames(timeout_ms=1000)
                if self.realsense_align is not None:
                    frames = self.realsense_align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    if depth_frame:
                        raw_depth = np.asanyarray(depth_frame.get_data())
                        if (raw_depth.shape[1], raw_depth.shape[0]) != (target_w, target_h):
                            self.latest_depth_array = cv2.resize(raw_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            self.latest_depth_array = raw_depth
                    else:
                        self.latest_depth_array = None

                    if (color_img.shape[1], color_img.shape[0]) != (target_w, target_h):
                        color_img = cv2.resize(color_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    self.frame_width = target_w
                    self.frame_height = target_h
                    frame = color_img
            except Exception as e:
                print(f"[RealSense] Blad odczytu klatki: {e}")

        # 2. Tradycyjna kamera USB
        elif not self.use_simulation and self.cap is not None and self.cap.isOpened():
            self.latest_depth_array = None
            ret, f = self.cap.read()
            if ret and f is not None:
                if (f.shape[1], f.shape[0]) != (target_w, target_h):
                    f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                self.frame_width = target_w
                self.frame_height = target_h
                frame = f

        # 3. Tryb symulacyjny (ladowanie testowego zdjecia laki)
        if frame is None:
            self.latest_depth_array = None
            test_images = [
                PROJECT_ROOT / "data" / "external_test_images" / "ragwort_field_1.jpg",
                PROJECT_ROOT / "data" / "external_test_images" / "ragwort_flowers_2.jpg",
                PROJECT_ROOT / "data" / "combined_dataset" / "test" / "images" / "jkk_jkk0001_jpg.rf.7927955a370eff71ccacce7dd950e3c1.jpg",
            ]
            for t in test_images:
                if t.exists():
                    f = cv2.imread(str(t))
                    if f is not None:
                        frame = cv2.resize(f, (target_w, target_h))
                        break

            if frame is None:
                dummy = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                dummy[:] = (30, 60, 30)  # Dark green meadow
                cv2.putText(dummy, f"Camera Simulation ({target_w}x{target_h})", (60, target_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                frame = dummy

            self.frame_width = target_w
            self.frame_height = target_h

        # Programowa korekcja jasnosci/kontrastu dla trybu symulacji lub kamer bez wsparcia ISP
        if self.camera_mode != "realsense" and frame is not None:
            alpha = max(0.2, float(self.camera_settings.get("contrast", 50)) / 50.0)
            beta = float(self.camera_settings.get("brightness", 0))
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Adaptacyjna normalizacja oswietlenia CLAHE (stabilizacja slonce/cien)
        if self.camera_settings.get("clahe_enabled", False) and frame is not None:
            clip = float(self.camera_settings.get("clahe_clip", 2.0))
            frame = self.normalize_illumination(frame, clip_limit=clip)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Przetwarza pojedyncza klatke przez model i modul decyzyjny Opcji C.
        Zwraca: (obraz z nalozonymi maskami i HUD, wynik analizy organow, lista detekcji dla robota).
        """
        if self.model is None:
            return frame, self.latest_analysis, []

        h, w = frame.shape[:2]
        predict_kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        # Przekazanie rozdzielczosci inferencji (imgsz) dla modeli YOLO
        if hasattr(self, "inference_imgsz") and self.inference_imgsz:
            predict_kwargs["imgsz"] = self.inference_imgsz

        try:
            res = self.model.predict(frame, **predict_kwargs)[0]
        except TypeError:
            # Fallback dla modeli nieobslugujacych imgsz
            predict_kwargs.pop("imgsz", None)
            res = self.model.predict(frame, **predict_kwargs)[0]

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

        # 2. Renderowanie masek i wizualizacji (skalowane proporcjonalnie do 1080p)
        overlay = frame.copy()
        detections: List[Dict[str, Any]] = []

        is_hd = w >= 1280
        line_thick = 3 if is_hd else 2
        font_scale = 0.65 if is_hd else 0.5
        cross_size = 14 if is_hd else 10
        circle_radius = 5 if is_hd else 4

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
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=line_thick)

            # Rysowanie ramki i etykiety
            if len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2

                # Pomiar odleglosci za pomoca mapy glebi RealSense D405
                dist_cm = None
                if getattr(self, "latest_depth_array", None) is not None:
                    dh, dw = self.latest_depth_array.shape[:2]
                    if 0 <= yc < dh and 0 <= xc < dw:
                        patch = self.latest_depth_array[max(0, yc - 4):min(dh, yc + 5), max(0, xc - 4):min(dw, xc + 5)]
                        valid = patch[patch > 0]
                        if len(valid) > 0:
                            dist_cm = round(float(np.median(valid)) / 10.0, 1)  # mm -> cm

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thick)
                # Punkt celowniczy dla manipulatora robota Hege
                cv2.circle(frame, (xc, yc), circle_radius, (0, 255, 255), -1)
                cv2.drawMarker(frame, (xc, yc), (0, 255, 255), cv2.MARKER_CROSS, cross_size, 2)

                label_txt = f"{cname} {score*100:.0f}%"
                if dist_cm is not None:
                    label_txt += f" [{dist_cm}cm]"

                cv2.putText(frame, label_txt, (x1, max(24 if is_hd else 18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2 if is_hd else 1, cv2.LINE_AA)

                det_dict = {
                    "class": cname,
                    "confidence": round(float(score), 2),
                    "box": [x1, y1, x2, y2],
                    "center": [xc, yc],
                }
                if dist_cm is not None:
                    det_dict["distance_cm"] = dist_cm
                detections.append(det_dict)

        # Mieszanie masek poligonowych (alpha blending)
        if len(masks_xy) > 0:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 3. Pasek naglowkowy HUD (Wskaznik Opcji C / DEIM i rozdzielczosci)
        hud_height = 54 if is_hd else 42
        hud = np.zeros((hud_height, w, 3), dtype=np.uint8)

        model_label = getattr(self, "model_type", "YOLOv8")
        if "DEIM" in model_label:
            tag = "DEIMv2 + DINOv3"
        elif self.is_segmentation:
            tag = "SEGMENTATION (OPTION C)"
        else:
            tag = "YOLOv8 DETECTION"

        sensor_tag = "RealSense D405 RGB-D" if self.camera_mode == "realsense" else f"CAM {w}x{h}"

        if analysis["is_ragwort"]:
            hud[:] = (15, 110, 25)  # Dark green
            verdict_str = f"VERDICT: COMMON RAGWORT ({analysis['confidence']*100:.0f}%) | {tag} [{sensor_tag}]"
        else:
            hud[:] = (20, 20, 110)  # Dark red
            verdict_str = f"VERDICT: NO RAGWORT / OTHER VEGETATION | {tag} [{sensor_tag}]"

        counts = analysis["counts"]
        if counts.get("ragwort", 0) > 0 and (counts["flowers"] + counts["leaves"] + counts["stems"] == 0):
            details_str = f"Detected: {counts['ragwort']} ragwort plants (Model: {self.model_path.name})"
        else:
            details_str = f"Organs: {counts['flowers']} flowers | {counts['leaves']} leaves | {counts['stems']} stems"

        fs_title = 0.65 if is_hd else 0.55
        fs_det = 0.48 if is_hd else 0.42
        cv2.putText(hud, verdict_str, (14, 24 if is_hd else 18), cv2.FONT_HERSHEY_SIMPLEX, fs_title, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, details_str, (14, 46 if is_hd else 35), cv2.FONT_HERSHEY_SIMPLEX, fs_det, (220, 220, 220), 1, cv2.LINE_AA)

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

        # Watermark FPS & resolution in bottom-right corner
        h, w = processed_frame.shape[:2]
        res_tag = pipeline.camera_resolution.upper() if pipeline.camera_resolution else ""
        fps_text = f"FPS: {pipeline.fps} | {pipeline.frame_width}x{pipeline.frame_height} ({res_tag})"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        tx = max(10, w - tw - 16)
        ty = h - 14
        # Pill backdrop with cyan accent border
        cv2.rectangle(processed_frame, (tx - 8, ty - th - 6), (tx + tw + 8, ty + 6), (15, 23, 42), -1)
        cv2.rectangle(processed_frame, (tx - 8, ty - th - 6), (tx + tw + 8, ty + 6), (56, 189, 248), 1)
        cv2.putText(processed_frame, fps_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
        model_family=getattr(pipeline, "model_family", "yolo"),
        is_segmentation=pipeline.is_segmentation,
        model_type=getattr(pipeline, "model_type", "YOLOv8"),
        conf_default=pipeline.conf_threshold,
        iou_default=pipeline.iou_threshold,
        camera_resolution=pipeline.camera_resolution,
        resolution=f"{pipeline.frame_width}x{pipeline.frame_height}",
        inference_imgsz=pipeline.inference_imgsz,
        camera_mode=pipeline.camera_mode,
        realsense_name=pipeline.realsense_device_name,
        camera_settings=pipeline.camera_settings,
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
        "model_family": getattr(pipeline, "model_family", "yolo"),
        "is_segmentation": pipeline.is_segmentation,
        "model_type": getattr(pipeline, "model_type", "YOLOv8"),
        "resolution": f"{pipeline.frame_width}x{pipeline.frame_height}",
        "camera_resolution": pipeline.camera_resolution,
        "inference_imgsz": pipeline.inference_imgsz,
        "camera_mode": pipeline.camera_mode,
        "realsense_available": (pipeline.realsense_device_name is not None),
        "realsense_name": pipeline.realsense_device_name or "None",
        "has_depth": (pipeline.latest_depth_array is not None),
        "camera_settings": pipeline.camera_settings,
        "analysis": pipeline.latest_analysis,
        "detections": pipeline.latest_detections,
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    """Zmiana progow detekcji oraz zrodla i rozdzielczosci kamery w locie."""
    data = request.json or {}
    if "conf" in data:
        pipeline.conf_threshold = float(data["conf"])
    if "iou" in data:
        pipeline.iou_threshold = float(data["iou"])
    if "inference_imgsz" in data:
        pipeline.inference_imgsz = int(data["inference_imgsz"])
    if "camera_resolution" in data:
        res = str(data["camera_resolution"])
        if res in pipeline.RESOLUTIONS:
            pipeline.set_camera_resolution(res)
    if "camera_mode" in data:
        mode = str(data["camera_mode"])
        pipeline.init_camera(mode=mode, resolution=pipeline.camera_resolution)

    return jsonify({
        "status": "success",
        "conf": pipeline.conf_threshold,
        "iou": pipeline.iou_threshold,
        "inference_imgsz": pipeline.inference_imgsz,
        "camera_resolution": pipeline.camera_resolution,
        "camera_mode": pipeline.camera_mode,
        "realsense_name": pipeline.realsense_device_name or "None",
        "resolution": f"{pipeline.frame_width}x{pipeline.frame_height}",
        "fps": pipeline.fps,
        "use_simulation": pipeline.use_simulation,
    })


@app.route("/api/camera_settings", methods=["GET", "POST"])
def api_camera_settings():
    """Pobiera lub modyfikuje parametry sprzetowe ISP RealSense D405 oraz normalizacji."""
    if request.method == "POST":
        data = request.json or {}
        updated = pipeline.apply_camera_settings(data)
        return jsonify({"status": "success", "settings": updated})
    return jsonify({
        "status": "success",
        "settings": pipeline.camera_settings,
        "is_realsense": (pipeline.camera_mode == "realsense"),
        "realsense_name": pipeline.realsense_device_name or "Brak",
    })


@app.route("/api/camera_settings/preset", methods=["POST"])
def api_camera_preset():
    """Stosuje gotowy profil oswietleniowy dla warunkow polowych."""
    data = request.json or {}
    preset = data.get("preset", "reset")

    if preset == "sun":
        # Profil: Ostre slonce na lace (redukcja przepalen, wlaczone CLAHE i kompensacja)
        settings = {
            "auto_exposure": True,
            "auto_white_balance": True,
            "contrast": 60,
            "brightness": -12,
            "saturation": 55,
            "gamma": 280,
            "sharpness": 60,
            "backlight_compensation": True,
            "clahe_enabled": True,
            "clahe_clip": 2.5,
        }
    elif preset == "cloud":
        # Profil: Cien / Chmury (podciagniecie ciemnych partii i cieplejsza barwa)
        settings = {
            "auto_exposure": True,
            "auto_white_balance": True,
            "contrast": 50,
            "brightness": 12,
            "saturation": 70,
            "gamma": 360,
            "sharpness": 55,
            "backlight_compensation": False,
            "clahe_enabled": True,
            "clahe_clip": 2.0,
        }
    elif preset == "invariant":
        # Profil: Maksymalna stabilizacja (sztuczne zrownowazenie slonca i cienia)
        settings = {
            "auto_exposure": True,
            "auto_white_balance": True,
            "contrast": 50,
            "brightness": 0,
            "saturation": 60,
            "gamma": 300,
            "sharpness": 50,
            "backlight_compensation": True,
            "clahe_enabled": True,
            "clahe_clip": 3.0,
        }
    else:  # "reset"
        settings = {
            "auto_exposure": True,
            "exposure": 33000,
            "gain": 16,
            "auto_white_balance": True,
            "white_balance": 4600,
            "contrast": 50,
            "brightness": 0,
            "saturation": 64,
            "gamma": 300,
            "sharpness": 50,
            "backlight_compensation": False,
            "clahe_enabled": False,
            "clahe_clip": 2.0,
        }

    updated = pipeline.apply_camera_settings(settings)
    return jsonify({"status": "success", "preset": preset, "settings": updated})


@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Lista dostepnych modeli w projekcie z wyraznym podzialem na rodziny: YOLO i DEIM."""
    yolo_models = []
    deim_models = []
    all_models = []
    seen_names = set()

    # Szukanie we wszystkich folderach modeli i wag (.pt oraz .pth)
    for folder in [MODELS_DIR, WEIGHTS_DIR, APP_DIR / "weights", PROJECT_ROOT]:
        if folder.exists():
            weight_files = list(folder.glob("*.pt")) + list(folder.glob("*.pth"))
            for f in sorted(weight_files, key=lambda x: x.name.lower()):
                if f.name in seen_names:
                    continue
                seen_names.add(f.name)
                size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                name_l = f.name.lower()

                is_deim = ("deim" in name_l) or (f.name == "model.pt") or name_l.endswith(".pth")
                if is_deim:
                    is_pth = name_l.endswith(".pth")
                    entry = {
                        "name": f.name,
                        "path": str(f),
                        "size_mb": size_mb,
                        "extension": f.suffix.lower(),
                        "family": "deim",
                        "type": "deim",
                        "tag": "DEIMv2 + DINOv3 (PyTorch .pth)" if is_pth else "DEIMv2 + DINOv3 Transformer",
                        "description": "PyTorch training checkpoint (.pth)" if is_pth else "Vision Transformer detector based on DINOv3 visual representations",
                        "recommended": f.name in ["model.pt", "deim_dinov3_ragwort.pt", "checkpoint0044.pth"],
                    }
                    deim_models.append(entry)
                    all_models.append(entry)
                else:
                    is_seg = ("seg" in name_l)
                    if is_seg:
                        mtype = "segment"
                        tag = "Organ Segmentation (Option C)"
                        desc = "Precise instance segmentation: flowers, leaves, stems"
                    elif "8s" in name_l or "best" in name_l or (18 * 1024 * 1024 < f.stat().st_size < 100 * 1024 * 1024):
                        mtype = "detect"
                        tag = "YOLOv8s Weed Detection"
                        desc = "Full ragwort plant detection using bounding boxes"
                    elif "8n" in name_l:
                        mtype = "detect"
                        tag = "YOLOv8n Weed Detection"
                        desc = "Lightweight YOLOv8 nano model for high-FPS detection"
                    else:
                        mtype = "detect"
                        tag = "YOLOv8 Detection"
                        desc = "YOLOv8 bounding box detection model"

                    entry = {
                        "name": f.name,
                        "path": str(f),
                        "size_mb": size_mb,
                        "extension": f.suffix.lower(),
                        "family": "yolo",
                        "type": mtype,
                        "tag": tag,
                        "description": desc,
                        "recommended": f.name in [
                            "ragwort_segmentation_best.pt",
                            "ragwort_yolov8s_best.pt",
                            "ragwort_yolov8_seg_best.pt",
                            "best (1).pt"
                        ],
                    }
                    yolo_models.append(entry)
                    all_models.append(entry)

    # Sort: recommended first, segmentation before detection
    yolo_models.sort(key=lambda x: (not x["recommended"], x["type"] != "segment", x["name"]))
    deim_models.sort(key=lambda x: (not x["recommended"], x["name"]))

    return jsonify({
        "active_model": pipeline.model_path.name,
        "active_family": getattr(pipeline, "model_family", "yolo"),
        "is_segmentation": pipeline.is_segmentation,
        "model_type": getattr(pipeline, "model_type", "YOLOv8"),
        "families": {
            "yolo": yolo_models,
            "deim": deim_models
        },
        "models": all_models
    })


@app.route("/api/models/switch", methods=["POST"])
def api_switch_model():
    """Switch active model and/or architecture (YOLO vs DEIM) in the vision pipeline."""
    data = request.json or {}
    target = data.get("model")
    family = data.get("family")  # 'yolo' or 'deim'

    # If only family is provided without specific weight filename:
    if not target and family:
        if family == "deim":
            for cand in [
                WEIGHTS_DIR / "checkpoint0044.pth",
                WEIGHTS_DIR / "model.pt",
                MODELS_DIR / "deim_dinov3_ragwort.pt"
            ]:
                if cand.exists():
                    target = str(cand)
                    break
        elif family == "yolo":
            for cand in [
                MODELS_DIR / "ragwort_segmentation_best.pt",
                MODELS_DIR / "ragwort_yolov8_seg_best.pt",
                MODELS_DIR / "ragwort_yolov8s_best.pt",
                WEIGHTS_DIR / "best (1).pt",
            ]:
                if cand.exists():
                    target = str(cand)
                    break

    if not target:
        return jsonify({"status": "error", "message": "Missing model or family parameter"}), 400

    target_path = Path(target)
    if not target_path.exists():
        for folder in [MODELS_DIR, WEIGHTS_DIR, APP_DIR / "weights", PROJECT_ROOT]:
            candidate = folder / target
            if candidate.exists():
                target_path = candidate
                break
            candidate_name = folder / Path(target).name
            if candidate_name.exists():
                target_path = candidate_name
                break

    success = pipeline.load_model(target_path, force_family=family)
    return jsonify({
        "status": "success" if success else "error",
        "active_model": pipeline.model_path.name,
        "active_family": getattr(pipeline, "model_family", "yolo"),
        "is_segmentation": pipeline.is_segmentation,
        "model_type": getattr(pipeline, "model_type", "YOLOv8"),
    })


@app.route("/api/sample_images", methods=["GET"])
def api_sample_images():
    """Returns list of sample meadow images with metadata."""
    samples_dir = PROJECT_ROOT / "data" / "external_test_images"
    samples = []
    titles = {
        "ragwort_field_1.jpg": "🌾 Ragwort Meadow (Field)",
        "ragwort_flowers_2.jpg": "🌼 Ragwort Inflorescence (Close-up)",
        "negative_dandelion_3.jpg": "🌻 Dandelion (Negative / Similar weed)",
        "negative_meadow_4.jpg": "🌿 Grassland Meadow (Negative)",
    }
    if samples_dir.exists():
        for p in sorted(samples_dir.glob("*.jpg")):
            samples.append({
                "filename": p.name,
                "title": titles.get(p.name, p.stem.replace("_", " ").title()),
                "is_ragwort": "ragwort" in p.name.lower(),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            })
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/process_sample", methods=["POST"])
def api_process_sample():
    """Processes selected sample image through active AI model (YOLO / DEIM)."""
    data = request.json or {}
    sample_name = data.get("filename")
    if not sample_name:
        return jsonify({"status": "error", "message": "Missing filename parameter"}), 400

    target = PROJECT_ROOT / "data" / "external_test_images" / Path(sample_name).name
    if not target.exists():
        return jsonify({"status": "error", "message": "Sample file not found"}), 404

    image = cv2.imread(str(target))
    if image is None:
        return jsonify({"status": "error", "message": "Failed to decode image file"}), 500

    # Resize if huge to prevent memory exhaustion
    h, w = image.shape[:2]
    if max(h, w) > 2560:
        scale = 2560.0 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    processed_img, analysis, detections = pipeline.process_frame(image)

    _, buffer = cv2.imencode(".jpg", processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "status": "success",
        "sample_name": sample_name,
        "analysis": analysis,
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
    })


@app.route("/api/process_image", methods=["POST"])
def api_process_image():
    """
    Endpoint for processing single user-uploaded image.
    Executes full Option C / DEIM pipeline and returns processed Base64 image and statistics.
    """
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"status": "error", "message": "Failed to decode image file"}), 400

    processed_img, analysis, detections = pipeline.process_frame(image)

    # Convert result to JPEG Base64 for browser
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
    Generates ready-to-run training command for external GPU machine / Colab.
    """
    data = request.json or {}
    model_name = data.get("model", "yolov8s.pt")
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

    colab_script = f"""# --- Run on Google Colab (GPU) ---
!git clone <YOUR_REPOSITORY_URL>
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
# Server Startup
# --------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"===========================================================")
    print(f"  🌿 Ragwort Detection & Vision Intelligence Dashboard")
    print(f"  URL: http://localhost:{port}")
    print(f"  Active model: {pipeline.model_path.name}")
    print(f"  Engine: {getattr(pipeline, 'model_type', 'YOLOv8')}")
    print(f"===========================================================")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
