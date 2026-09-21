import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from io import BytesIO


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/backgrounds")


IMAGES_PER_CATEGORY = 500


GBIF_LIMIT = 300


MIN_WIDTH = 800
MIN_HEIGHT = 600


REQUEST_TIMEOUT = 30


REQUEST_DELAY = 0.2


# ============================================================
# BACKGROUND CATEGORIES
# ============================================================

SEARCHES = {
    "grass": [
        "grassland",
        "grass field",
        "grass meadow",
    ],

    "meadow": [
        "meadow",
        "wildflower meadow",
        "natural meadow",
        "pasture",
    ],

    "field": [
        "field landscape",
        "agricultural field",
        "pasture field",
        "farmland",
    ],

    "roadside": [
        "roadside vegetation",
        "roadside grass",
        "roadside meadow",
    ],

    "soil": [
        "field soil",
        "bare soil field",
        "agricultural soil",
    ],
}


# ============================================================
# DIRECTORIES
# ============================================================

for category in SEARCHES:
    (OUTPUT_DIR / category).mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RagwortBackgroundDataset/1.0"
})


# ============================================================
# HELPERS
# ============================================================

def safe_filename(text: str) -> str:
    
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"

    return "".join(
        c if c in allowed else "_"
        for c in text
    )


def image_hash(image_bytes: bytes) -> str:

    return hashlib.sha256(image_bytes).hexdigest()


def download_image(url: str):

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get("Content-Type", "")

        if "image" not in content_type:
            return None

        image_bytes = response.content

        if len(image_bytes) < 20_000:
            return None

        image = Image.open(BytesIO(image_bytes))

        image.load()

        width, height = image.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return None

        # RGB/RGBA
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        return image, image_bytes

    except Exception:
        return None


def save_image(image, category, index):


    filename = f"{category}_{index:05d}.jpg"

    path = OUTPUT_DIR / category / filename

    image.convert("RGB").save(
        path,
        "JPEG",
        quality=95,
    )

    return path


# ============================================================
# GBIF SEARCH
# ============================================================

def search_gbif(query, limit=300, offset=0):

    url = "https://api.gbif.org/v1/occurrence/search"

    params = {
        "q": query,
        "media_type": "StillImage",
        "limit": limit,
        "offset": offset,
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        print(f"[ERROR] GBIF request failed: {e}")

        return None


# ============================================================
# DOWNLOAD CATEGORY
# ============================================================

def download_category(category, queries):

    print()
    print("=" * 70)
    print(f"CATEGORY: {category}")
    print("=" * 70)

    output_dir = OUTPUT_DIR / category

    metadata_path = output_dir / "metadata.jsonl"

    downloaded = 0
    offset = 0

    hashes = set()

    metadata_file = open(
        metadata_path,
        "a",
        encoding="utf-8",
    )

    for query in queries:

        if downloaded >= IMAGES_PER_CATEGORY:
            break

        print()
        print(f"[GBIF] Searching: {query}")

        offset = 0

        while downloaded < IMAGES_PER_CATEGORY:

            data = search_gbif(
                query,
                limit=GBIF_LIMIT,
                offset=offset,
            )

            if not data:
                break

            results = data.get("results", [])

            if not results:
                break

            print(
                f"[GBIF] query='{query}' "
                f"offset={offset} "
                f"records={len(results)}"
            )

            for record in results:

                if downloaded >= IMAGES_PER_CATEGORY:
                    break

                gbif_id = record.get("key")

                media = record.get("media", [])

                if not media:
                    continue

                # ------------------------------------------------
                # find first image
                # ------------------------------------------------

                image_url = None

                for m in media:

                    url = (
                        m.get("identifier")
                        or m.get("references")
                    )

                    if url:
                        image_url = url
                        break

                if not image_url:
                    continue

                # ------------------------------------------------
                # download
                # ------------------------------------------------

                result = download_image(image_url)

                if result is None:
                    continue

                image, image_bytes = result

                # ------------------------------------------------
                # duplicate detection
                # ------------------------------------------------

                img_hash = image_hash(image_bytes)

                if img_hash in hashes:
                    continue

                hashes.add(img_hash)

                # ------------------------------------------------
                # save
                # ------------------------------------------------

                path = save_image(
                    image,
                    category,
                    downloaded,
                )

                # ------------------------------------------------
                # metadata
                # ------------------------------------------------

                metadata = {
                    "gbif_id": gbif_id,
                    "category": category,
                    "query": query,
                    "image_url": image_url,
                    "local_path": str(path),
                    "width": image.width,
                    "height": image.height,
                    "license": record.get("license"),
                    "creator": record.get("creator"),
                    "publisher": record.get("publisher"),
                    "dataset_name": record.get("datasetName"),
                    "dataset_key": record.get("datasetKey"),
                    "scientific_name": record.get("scientificName"),
                    "occurrence_status": record.get("occurrenceStatus"),
                    "country": record.get("country"),
                    "event_date": record.get("eventDate"),
                }

                metadata_file.write(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                metadata_file.flush()

                downloaded += 1

                print(
                    f"  [{downloaded:04d}/{IMAGES_PER_CATEGORY}] "
                    f"{path.name}"
                )

                time.sleep(REQUEST_DELAY)

            offset += GBIF_LIMIT

            # GBIF search ma limit offsetu ~100k
            if offset >= 100_000:
                break

    metadata_file.close()

    print()
    print(
        f"[DONE] {category}: "
        f"{downloaded} images"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAGWORT BACKGROUND DATASET DOWNLOADER")
    print("=" * 70)

    print()
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images/category: {IMAGES_PER_CATEGORY}")

    for category, queries in SEARCHES.items():

        download_category(
            category,
            queries,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()