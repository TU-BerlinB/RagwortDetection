"""
File: src/dataset/dataset_download.py
Usage:
    python src/dataset/dataset_download.py
Description:
    Downloads sample images in parallel from URLs specified in a dataset CSV
    file using ThreadPoolExecutor and saves them into an images folder.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATASET_CSV = "sample_10000.csv"
OUT_DIR = Path("images")
OUT_DIR.mkdir(exist_ok=True)


def download_image(row):
    try:
        url = row["identifier"]
        gbif_id = row["gbifID"]

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            filename = OUT_DIR / f"{gbif_id}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            return f"Downloaded {gbif_id}"

    except Exception:
        return None


def main():
    if not Path(DATASET_CSV).exists():
        print(f"File not found: {DATASET_CSV}")
        return

    df = pd.read_csv(DATASET_CSV)
    rows = [row for _, row in df.head(10).iterrows()]

    with ThreadPoolExecutor(max_workers=20) as executor:
        for result in executor.map(download_image, rows):
            if result:
                print(result)

    print("Done!")


if __name__ == "__main__":
    main()