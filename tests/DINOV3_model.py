import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

image = Image.new("RGB", (224, 224), "green")

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print("Model loaded successfully!")
print("Output shape:", outputs.last_hidden_state.shape)