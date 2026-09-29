import sys
import tempfile
from pathlib import Path

# Dodanie katalogu głównego projektu do ścieżki wyszukiwania modułów
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from src.models.yolov8 import YOLOv8
from src.models.model import YOLOv8 as YOLOv8FromModel
from src.models import YOLOv8 as YOLOv8FromInit


def test_imports():
    assert YOLOv8 is YOLOv8FromModel
    assert YOLOv8 is YOLOv8FromInit
    print("[OK] Importy z src.models działają poprawnie!")


def test_inference():
    model = YOLOv8("yolov8s.pt")
    print(f"Device: {model.device}")

    # Test na obrazie PIL
    image = Image.new("RGB", (640, 640), "green")
    preds = model.predict(image, conf=0.25)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "scores" in preds[0]
    assert "labels" in preds[0]
    assert "class_names" in preds[0]
    print("[OK] Predykcja na obrazie PIL zakończona sukcesem!")

    # Test wywołania jako funkcji model(image)
    preds_call = model(image)
    assert len(preds_call) == 1
    print("[OK] Wywołanie model(image) działa!")

    # Test na tablicy numpy
    np_image = np.zeros((480, 640, 3), dtype=np.uint8)
    preds_np = model.predict(np_image)
    assert preds_np[0]["orig_shape"] == (480, 640)
    print("[OK] Predykcja na tablicy NumPy działa poprawnie!")


def test_predict_for_eval():
    model = YOLOv8("yolov8s.pt")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Tworzymy 2 tymczasowe zdjęcia
        img1 = Image.new("RGB", (300, 300), "red")
        img1.save(tmp_dir / "test_1.jpg")
        img2 = Image.new("RGB", (300, 300), "blue")
        img2.save(tmp_dir / "test_2.jpg")

        eval_preds = model.predict_for_eval(tmp_dir)

        assert "test_1.jpg" in eval_preds
        assert "test_2.jpg" in eval_preds
        assert isinstance(eval_preds["test_1.jpg"], list)
        print("[OK] predict_for_eval() dla data_eval.py działa poprawnie!")


def main():
    print("=== Rozpoczynam testy YOLOv8 ===")
    test_imports()
    test_inference()
    test_predict_for_eval()
    print("=== Wszystkie testy YOLOv8 zakończone SUKCESEM! ===")


if __name__ == "__main__":
    main()
