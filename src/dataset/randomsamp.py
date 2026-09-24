"""
File: src/dataset/randomsamp.py
Usage:
    python src/dataset/randomsamp.py
Description:
    Reads multimedia metadata from multimedia.txt, randomly samples 10,000 records
    with a fixed seed, and saves the sample to sample_10000.csv.
"""

from pathlib import Path
import pandas as pd

MULTIMEDIA_PATH = Path("src/dataset/multimedia.txt")
OUTPUT_PATH = Path("sample_10000.csv")

if MULTIMEDIA_PATH.exists():
    df = pd.read_csv(MULTIMEDIA_PATH, sep="\t")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")

    small_sample = df.sample(n=min(10000, len(df)), random_state=42)
    small_sample.to_csv(
        OUTPUT_PATH,
        index=False,
    )
    print(f"Saved {len(small_sample)} sample records to {OUTPUT_PATH}")
else:
    print(f"Input file not found: {MULTIMEDIA_PATH}")