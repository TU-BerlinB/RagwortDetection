import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Load data
df = pd.read_csv("sample_100.csv")

# Images folder
out_dir = Path("images_small")
out_dir.mkdir(exist_ok=True)

def download_image(row):
    try:
        url = row["identifier"]
        gbif_id = row["gbifID"]

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            filename = out_dir / f"{gbif_id}.jpg"

            with open(filename, "wb") as f:
                f.write(response.content)

            return f"Downloaded {gbif_id}"

    except Exception:
        return None

# Download first 10000 rows
rows = [row for _, row in df.head(10000).iterrows()]

with ThreadPoolExecutor(max_workers=20) as executor:
    for result in executor.map(download_image, rows):
        if result:
            print(result)

print("Done!")