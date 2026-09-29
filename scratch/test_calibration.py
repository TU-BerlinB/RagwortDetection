import requests
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
import torch
import numpy as np

clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

det_proc = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
det_model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

r = requests.get("https://api.gbif.org/v1/occurrence/search?taxonKey=5388602&mediaType=StillImage&limit=25").json()

prompts = [
    "a macro close-up of a flower or plant filling the frame",
    "a wild plant growing in a grassy meadow or field from a distance",
]

passed_count = 0
results_list = r.get("results", [])
print(f"Testing {len(results_list)} occurrences...")

for occ in results_list:
    occ_id = occ["key"]
    media = occ.get("media", [])
    if not media:
        continue
    url = media[0].get("identifier")
    if not url:
        continue

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.status_code != 200 or len(resp.content) < 1500:
            continue
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        continue

    w, h = img.size

    # 1. CLIP
    inputs = clip_proc(text=prompts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
    p_macro = float(probs[0].item())
    p_meadow = float(probs[1].item())

    # 2. OWLv2
    inputs = det_proc(text=[["yellow flower plant", "ragwort plant"]], images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = det_model(**inputs)
    results = det_proc.post_process_grounded_object_detection(outputs=outputs, target_sizes=[(h, w)], threshold=0.15)[0]
    
    plant_mask = np.zeros((h, w), dtype=bool)
    max_h_ratio = 0.0
    for b in results["boxes"]:
        x1, y1, x2, y2 = [int(v.item()) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            plant_mask[y1:y2, x1:x2] = True
            box_h = (y2 - y1) / float(h)
            if box_h > max_h_ratio:
                max_h_ratio = box_h
    coverage = float(np.sum(plant_mask)) / (w * h)

    # 3. Depth
    depth_out = depth_pipe(img)
    depth_map = np.array(depth_out["depth"])
    if depth_map.shape != (h, w):
        depth_map = np.array(Image.fromarray(depth_map).resize((w, h), Image.BILINEAR))
    
    plant_depths = depth_map[plant_mask] if np.any(plant_mask) else depth_map
    mean_d = float(np.mean(plant_depths))
    p90_d = float(np.percentile(plant_depths, 90))

    # Evaluate filters
    clip_pass = p_meadow >= p_macro
    area_pass = coverage < 0.30  # plant does not take up entire screen / < 30%
    depth_pass = (mean_d < 175.0) and (p90_d < 215.0)

    status = "PASSED" if (clip_pass and area_pass and depth_pass) else "REJECTED"
    reason = []
    if not clip_pass: reason.append(f"CLIP Macro ({p_macro*100:.0f}%)")
    if not area_pass: reason.append(f"Area ({coverage*100:.0f}%>=30%)")
    if not depth_pass: reason.append(f"Depth (mean={mean_d:.0f}, p90={p90_d:.0f})")

    print(f"Occ #{occ_id}: {status} [{' ,'.join(reason) if reason else 'All passed'}] Coverage={coverage*100:.1f}%, MeanDepth={mean_d:.1f}")
    if status == "PASSED":
        passed_count += 1

print(f"\nFinal: {passed_count} / {len(results_list)} passed!")
