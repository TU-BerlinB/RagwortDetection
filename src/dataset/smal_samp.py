"""
File: src/dataset/smal_samp.py
Usage:
    python src/dataset/smal_samp.py
Description:
    Iterates through image URLs in sample_100.csv and opens each image URL
    sequentially in a web browser for manual visual inspection.
"""

from pathlib import Path
import pandas as pd
import webbrowser

CSV_PATH = Path("sample_100.csv")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    for i, url in enumerate(df["identifier"], start=1):
        print(f"Opening {i}/{len(df)}: {url}")
        webbrowser.open(url)
        input("Press Enter for next image...")
else:
    print(f"File not found: {CSV_PATH}")
