"""
File: src/scripts/filter_backgrounds.py
Usage:
    python src/scripts/filter_backgrounds.py
Description:
    Filters background landscape images using SigLIP zero-shot vision-language embeddings,
    scoring candidate images against positive (wide meadow/field) and negative (macro/close-up)
    prompts, separating them into accepted and rejected directories.
"""

from pathlib import Path
import shutil

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

INPUT_DIR = Path("data/backgrounds")
OUTPUT_ACCEPTED = INPUT_DIR / "accepted"
OUTPUT_REJECTED = INPUT_DIR / "rejected"

MODEL_NAME = "google/siglip-base-patch16-224"
THRESHOLD = 0.00

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Positive prompts describing desired wide landscape images
POSITIVE_PROMPTS = [
    "a wide photograph of a meadow",
    "a wide photograph of a grass field",
    "a distant view of a grass field",
    "a landscape photograph of a meadow",
    "a natural meadow viewed from a distance",
    "a large grassy field viewed from a distance",
    "a wide outdoor field with grass",
    "a natural grassland landscape",
    "a photograph of a field with grass",
    "a photograph of a soil with plants",
]

# Negative prompts describing undesirable close-ups and insects
NEGATIVE_PROMPTS = [
    "a close-up photograph of grass",
    "a macro photograph of grass",
    "a close-up photograph of a grass plant",
    "a single grass plant filling the frame",
    "a single plant photographed from very close",
    "a detailed close-up of leaves",
    "a close-up of grass blades",
    "a macro photograph of leaves",
    "a plant filling almost the entire image",
    "wasp, bees, butterfly, moth",
]


def get_text_embeddings(processor, model, device, prompts):
    inputs = processor(
        text=prompts,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def calculate_score(processor, model, device, image, positive_embeddings, negative_embeddings):
    """Calculate positive score, negative score, and margin (positive - negative)."""
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    if hasattr(outputs, "pooler_output"):
        image_embedding = outputs.pooler_output
    else:
        image_embedding = outputs

    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    positive_scores = (image_embedding @ positive_embeddings.T)[0]
    negative_scores = (image_embedding @ negative_embeddings.T)[0]

    positive_score = positive_scores.mean().item()
    negative_score = negative_scores.mean().item()
    margin = positive_score - negative_score

    return positive_score, negative_score, margin


def get_images():
    images = []
    for path in INPUT_DIR.rglob("*"):
        if OUTPUT_ACCEPTED in path.parents or OUTPUT_REJECTED in path.parents:
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    positive_embeddings = get_text_embeddings(processor, model, device, POSITIVE_PROMPTS)
    negative_embeddings = get_text_embeddings(processor, model, device, NEGATIVE_PROMPTS)

    images = get_images()
    print(f"\nFound {len(images)} images to filter.")

    OUTPUT_ACCEPTED.mkdir(parents=True, exist_ok=True)
    OUTPUT_REJECTED.mkdir(parents=True, exist_ok=True)

    results = []
    accepted_count = 0
    rejected_count = 0

    for image_path in tqdm(images, desc="Filtering images"):
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            positive_score, negative_score, margin = calculate_score(
                processor, model, device, image, positive_embeddings, negative_embeddings
            )
            accepted = margin >= THRESHOLD

            try:
                relative_path = image_path.relative_to(INPUT_DIR)
            except ValueError:
                relative_path = Path(image_path.name)

            if accepted:
                destination = OUTPUT_ACCEPTED / relative_path
                accepted_count += 1
            else:
                destination = OUTPUT_REJECTED / relative_path
                rejected_count += 1

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)

            results.append({
                "file": str(image_path),
                "width": width,
                "height": height,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "margin": margin,
                "threshold": THRESHOLD,
                "status": "accepted" if accepted else "rejected",
            })
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            results.append({
                "file": str(image_path),
                "width": None,
                "height": None,
                "positive_score": None,
                "negative_score": None,
                "margin": None,
                "threshold": THRESHOLD,
                "status": "error",
            })

    df = pd.DataFrame(results)
    csv_path = INPUT_DIR / "filter_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("FILTER FINISHED")
    print("=" * 60)
    print(f"Total images:     {len(images)}")
    print(f"Accepted:         {accepted_count}")
    print(f"Rejected:         {rejected_count}")
    if len(images) > 0:
        print(f"Accepted ratio:   {accepted_count / len(images) * 100:.1f}%")
    print(f"Accepted images:  {OUTPUT_ACCEPTED}")
    print(f"Rejected images:  {OUTPUT_REJECTED}")
    print(f"Results CSV:      {csv_path}\n")


if __name__ == "__main__":
    main()