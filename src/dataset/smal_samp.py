import pandas as pd
import webbrowser

df = pd.read_csv("sample_100.csv")

for i, url in enumerate(df["identifier"], start=1):
    print(f"Opening {i}/{len(df)}")
    webbrowser.open(url)

    input("Press Enter for next image...")
