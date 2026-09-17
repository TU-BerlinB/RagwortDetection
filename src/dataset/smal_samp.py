import pandas as pd

df = pd.read_csv("sample_100.csv")

print(df.head())
print(len(df))

for url in df["identifier"].head():
    print(url)