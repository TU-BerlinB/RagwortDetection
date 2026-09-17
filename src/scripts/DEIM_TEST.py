import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


# Path to DEIMv2
DEIMV2_PATH = Path(__file__).resolve().parents[3] / "DEIMv2"
sys.path.insert(0, str(DEIMV2_PATH))
import sys

sys.path.insert(0, "/DEIMv2")


from engine.backbone.dinov3_adapter import DINOv3STAs


IMAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/dino_dataset/ragwort/"
    / "jkk_jkk0001_jpg.rf.7927955a370eff71ccacce7dd950e3c1.jpg"
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    image = Image.open(IMAGE_PATH).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    print(f"Input shape: {x.shape}")

    model = DINOv3STAs(
        name="dinov3",
        weights_path=None,
        interaction_indexes=[2, 5, 8],
        finetune=False,
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=256,
    )

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        features = model(x)

    print("\nDINOv3 / DEIMv2 features:")

    for i, feature in enumerate(features):
        print(f"Feature {i}: {feature.shape}")


if __name__ == "__main__":
    main()