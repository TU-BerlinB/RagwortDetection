from pathlib import Path
import shutil

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/backgrounds")

OUTPUT_ACCEPTED = INPUT_DIR / "accepted"
OUTPUT_REJECTED = INPUT_DIR / "rejected"

MODEL_NAME = "google/siglip-base-patch16-224"

# Im wyższy próg, tym bardziej rygorystyczny filtr.
THRESHOLD = 0.00

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Zdjęcia, które chcemy zachować
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
    "a photograph of a soil with plants"
]

# Zdjęcia, które chcemy odrzucać
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
    "wasp, bees, butterfly, moth"
]


# ============================================================
# MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading model: {MODEL_NAME}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# PROMPT EMBEDDINGS
# ============================================================

def get_text_embeddings(prompts):
    inputs = processor(
        text=prompts,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs

    embeddings = embeddings / embeddings.norm(
        dim=-1,
        keepdim=True
    )

    return embeddings

print("Calculating text embeddings...")

positive_embeddings = get_text_embeddings(POSITIVE_PROMPTS)
negative_embeddings = get_text_embeddings(NEGATIVE_PROMPTS)

print("Text embeddings ready.")


# ============================================================
# IMAGE SCORING
# ============================================================

def calculate_score(image):
    """
    Returns:

        positive_score
        negative_score
        margin
    """

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    if hasattr(outputs, "pooler_output"):
        image_embedding = outputs.pooler_output
    else:
        image_embedding = outputs

    image_embedding = image_embedding / image_embedding.norm(
        dim=-1,
        keepdim=True
    )

    positive_scores = (
        image_embedding @ positive_embeddings.T
    )[0]

    negative_scores = (
        image_embedding @ negative_embeddings.T
    )[0]

    positive_score = positive_scores.mean().item()
    negative_score = negative_scores.mean().item()

    margin = positive_score - negative_score

    return (
        positive_score,
        negative_score,
        margin
    )


# ============================================================
# FILES
# ============================================================

def get_images():

    images = []

    for path in INPUT_DIR.rglob("*"):

        # Don't process output directories
        if (
            OUTPUT_ACCEPTED in path.parents
            or OUTPUT_REJECTED in path.parents
        ):
            continue

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


images = get_images()

print()
print(f"Found {len(images)} images.")
print()


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_ACCEPTED.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_REJECTED.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILTER
# ============================================================

results = []

accepted_count = 0
rejected_count = 0


for image_path in tqdm(images, desc="Filtering images"):

    try:

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        positive_score, negative_score, margin = (
            calculate_score(image)
        )

        accepted = margin >= THRESHOLD

        # Preserve original category
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

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            image_path,
            destination
        )

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

        print(
            f"\nERROR processing {image_path}: {e}"
        )

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


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)

csv_path = INPUT_DIR / "filter_results.csv"

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FILTER FINISHED")
print("=" * 60)

print(f"Total images:     {len(images)}")
print(f"Accepted:         {accepted_count}")
print(f"Rejected:         {rejected_count}")

if len(images) > 0:

    print(
        f"Accepted ratio:   "
        f"{accepted_count / len(images) * 100:.1f}%"
    )

print()
print(f"Accepted images:  {OUTPUT_ACCEPTED}")
print(f"Rejected images:  {OUTPUT_REJECTED}")
print(f"Results CSV:      {csv_path}")
print()