import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# Same setup as training
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "top_view_classifier.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open(
    "./images/1976788122.jpg"
).convert("RGB")

img = transform(img).unsqueeze(0)

with torch.no_grad():

    outputs = model(img)

    prediction = torch.argmax(
        outputs,
        dim=1
    ).item()

classes = [
    "not_top_view",
    "top_view"
]

print(
    "Prediction:",
    classes[prediction]
)