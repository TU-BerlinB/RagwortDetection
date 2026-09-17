import pandas as pd
import requests
from pathlib import Path

# Load sample file
df = pd.read_csv("sample_10000.csv")

# Create images folder if it doesn't exist
out_dir = Path("images")
out_dir.mkdir(exist_ok=True)

# Only download 10 images for testing
N = 10

for idx, row in df.head(N).iterrows():

    url = row["identifier"]

    try:
        response = requests.get(url, timeout=20)

        if response.status_code == 200:

            filename = out_dir / f"{idx}.jpg"

            with open(filename, "wb") as f:
                f.write(response.content)

            print(f"Downloaded: {filename}")

        else:
            print(f"Failed ({response.status_code}): {url}")

    except Exception as e:
        print(f"Error: {e}")