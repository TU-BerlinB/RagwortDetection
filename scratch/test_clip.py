import requests
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

r = requests.get("https://api.gbif.org/v1/occurrence/search?taxonKey=5388602&mediaType=StillImage&limit=10").json()

prompts = [
    "a close-up macro photo of a flower or plant filling the frame",
    "a wild plant in a meadow or field with grass and natural surroundings",
]

for occ in r.get("results", []):
    occ_id = occ["key"]
    media = occ.get("media", [])
    if not media:
        continue
    url = media[0].get("identifier")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.status_code != 200 or len(resp.content) < 1500:
            continue
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        continue

    inputs = clip_proc(text=prompts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
    p_macro = float(probs[0].item())
    p_meadow = float(probs[1].item())
    decision = "MEADOW (PASS)" if p_meadow >= p_macro else "MACRO (DISCARD)"
    print(f"Occ #{occ_id}: Macro={p_macro*100:.1f}%, Meadow={p_meadow*100:.1f}% -> {decision}", flush=True)
