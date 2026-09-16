import torch
from PIL import Image

from src.models.dinov3 import DINOv3


def main():
    model = DINOv3()

    print(f"Device: {model.device}")

    image = Image.new("RGB", (224, 224), "green")

    features = model.extract_features(image)

    print("DINOv3 works!")
    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Feature device:", features.device)
    print("Feature min:", features.min().item())
    print("Feature max:", features.max().item())


if __name__ == "__main__":
    main()