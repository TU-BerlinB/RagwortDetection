from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

RAGWORT_IMAGES = Path(
    "data/combined_dataset/train/images"
)

RAGWORT_LABELS = Path(
    "data/combined_dataset/train/labels"
)

BACKGROUND_DIR = Path(
    "data/backgrounds/accepted"
)

OUTPUT_DIR = Path(
    "data/synthetic_dataset"
)

OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"


# Number of synthetic images
NUM_SYNTHETIC_IMAGES = 12000


# Random scale of the ragwort image
#
# 0.15 = ragwort image is 15% of background width
# 0.50 = ragwort image is 50% of background width
MIN_SCALE = 0.15
MAX_SCALE = 0.30


# Maximum number of ragworts per synthetic image
MIN_RAGWORTS = 1
MAX_RAGWORTS = 2


# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# Random seed for reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def get_images(directory):
    return [
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_yolo_labels(label_path):
    """
    Read YOLO labels.

    Returns list of:
        class_id, x_center, y_center, width, height
    """

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            labels.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return labels


def get_random_ragwort_crop():
    """
    Select random ragwort image and crop it
    using one of its YOLO bounding boxes.

    This means we don't paste the entire original
    image — only the region containing the ragwort.
    """

    while True:

        image_path = random.choice(
            ragwort_images
        )

        label_path = (
            RAGWORT_LABELS
            / f"{image_path.stem}.txt"
        )

        labels = read_yolo_labels(
            label_path
        )

        if not labels:
            continue

        # Pick one ragwort from the image
        label = random.choice(labels)

        class_id, xc, yc, bw, bh = label

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        # YOLO -> pixels

        x_center = xc * w
        y_center = yc * h

        box_width = bw * w
        box_height = bh * h

        x1 = int(
            x_center - box_width / 2
        )

        y1 = int(
            y_center - box_height / 2
        )

        x2 = int(
            x_center + box_width / 2
        )

        y2 = int(
            y_center + box_height / 2
        )

        # Clamp

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        return crop


def resize_ragwort(
    crop,
    background_width,
):
    """
    Resize ragwort crop to a random size
    relative to background width.
    """

    scale = random.uniform(
        MIN_SCALE,
        MAX_SCALE,
    )

    target_width = int(
        background_width * scale
    )

    crop_h, crop_w = crop.shape[:2]

    ratio = target_width / crop_w

    target_height = int(
        crop_h * ratio
    )

    resized = cv2.resize(
        crop,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def paste_ragwort(
    background,
    ragwort,
):
    """
    Paste ragwort at a random position with
    strong feathering around the entire crop.

    Returns:
        new background
        bbox
    """

    bg_h, bg_w = background.shape[:2]

    rh, rw = ragwort.shape[:2]

    # If too large, scale down

    if rw >= bg_w or rh >= bg_h:

        scale = min(
            (bg_w - 2) / rw,
            (bg_h - 2) / rh,
        )

        rw = max(1, int(rw * scale))
        rh = max(1, int(rh * scale))

        ragwort = cv2.resize(
            ragwort,
            (rw, rh),
            interpolation=cv2.INTER_AREA,
        )

    # Random position

    max_x = bg_w - rw

    x = random.randint(
        0,
        max_x,
    )

    max_y = bg_h - rh

    # Prefer lower parts of the image.
    # random.random() ** 0.5 -> more uniform
    # random.random() ** 0.3 -> stronger preference for bottom
    y_ratio = random.random() ** 0.35

    y = int(
        y_ratio * max_y
    )

    # ========================================================
    # STRONG FEATHERING MASK
    # ========================================================

    # Start with a solid mask
    mask = np.ones(
        (rh, rw),
        dtype=np.float32,
    )

    # Much stronger blur
    #
    # The larger this value,
    # the softer the transition.
    #
    # 51-101 gives a very wide transition.

    blur_size = random.choice([
        51,
        61,
        71,
        81,
        101,
    ])

    # Make sure kernel is odd
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(
        mask,
        (
            blur_size,
            blur_size,
        ),
        0,
    )

    # ========================================================
    # IMPORTANT:
    # Create an actual distance-based fade
    # ========================================================

    # Distance from every pixel to the nearest edge

    distance_x = np.minimum(
        np.arange(rw),
        np.arange(rw)[::-1],
    )

    distance_y = np.minimum(
        np.arange(rh),
        np.arange(rh)[::-1],
    )

    distance = np.minimum.outer(
        distance_y,
        distance_x,
    ).astype(
        np.float32
    )

    # Width of transition zone
    feather_width = random.randint(
        30,
        70,
    )

    # Smooth transition from 0 -> 1

    mask = np.clip(
        distance / feather_width,
        0.0,
        1.0,
    )

    # Smoothstep for more natural transition
    mask = (
        mask
        * mask
        * (
            3.0
            - 2.0 * mask
        )
    )

    # ========================================================
    # Blend
    # ========================================================

    background_region = background[
        y:y + rh,
        x:x + rw
    ].astype(
        np.float32
    )

    ragwort_float = ragwort.astype(
        np.float32
    )

    mask = mask[..., np.newaxis]

    blended = (
        ragwort_float * mask
        +
        background_region
        * (1.0 - mask)
    )

    blended = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )

    # ========================================================
    # Put result back
    # ========================================================

    background[
        y:y + rh,
        x:x + rw
    ] = blended

    return background, (
        x,
        y,
        rw,
        rh,
    )
def bbox_to_yolo(
    x,
    y,
    width,
    height,
    image_width,
    image_height,
):
    """
    Convert pixel bbox to YOLO format.
    """

    x_center = (
        x + width / 2
    ) / image_width

    y_center = (
        y + height / 2
    ) / image_height

    width_norm = (
        width / image_width
    )

    height_norm = (
        height / image_height
    )

    return (
        x_center,
        y_center,
        width_norm,
        height_norm,
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("SYNTHETIC RAGWORT DATASET")
print("=" * 60)

ragwort_images = get_images(
    RAGWORT_IMAGES
)

background_images = get_images(
    BACKGROUND_DIR
)

print(
    f"Ragwort images:    {len(ragwort_images)}"
)

print(
    f"Background images: {len(background_images)}"
)

print(
    f"Synthetic images:  {NUM_SYNTHETIC_IMAGES}"
)

print()

if not ragwort_images:
    raise RuntimeError(
        "No ragwort images found."
    )

if not background_images:
    raise RuntimeError(
        "No background images found."
    )


OUTPUT_IMAGES.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_LABELS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GENERATION
# ============================================================

for index in tqdm(
    range(NUM_SYNTHETIC_IMAGES),
    desc="Generating",
):

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    background_path = random.choice(
        background_images
    )

    background = cv2.imread(
        str(background_path),
        cv2.IMREAD_COLOR,
    )

    if background is None:
        continue

    bg_h, bg_w = background.shape[:2]

    synthetic = background.copy()

    annotations = []

    # --------------------------------------------------------
    # Number of ragworts
    # --------------------------------------------------------

    num_ragworts = random.randint(
        MIN_RAGWORTS,
        MAX_RAGWORTS,
    )

    # --------------------------------------------------------
    # Add ragworts
    # --------------------------------------------------------

    for _ in range(num_ragworts):

        crop = get_random_ragwort_crop()

        crop = resize_ragwort(
            crop,
            bg_w,
        )

        synthetic, bbox = paste_ragwort(
            synthetic,
            crop,
        )

        x, y, width, height = bbox

        (
            xc,
            yc,
            bw,
            bh,
        ) = bbox_to_yolo(
            x,
            y,
            width,
            height,
            bg_w,
            bg_h,
        )

        annotations.append(
            f"0 {xc:.6f} {yc:.6f} "
            f"{bw:.6f} {bh:.6f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"synthetic_{index:06d}.jpg"
    )

    image_path = (
        OUTPUT_IMAGES
        / filename
    )

    label_path = (
        OUTPUT_LABELS
        / filename.replace(
            ".jpg",
            ".txt",
        )
    )

    cv2.imwrite(
        str(image_path),
        synthetic,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(annotations)
        )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Images: {OUTPUT_IMAGES}"
)

print(
    f"Labels: {OUTPUT_LABELS}"
)