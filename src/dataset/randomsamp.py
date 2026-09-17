import pandas as pd

df = pd.read_csv("src\dataset\multimedia.txt", sep="\t")

print(df.shape)
print(df.columns)

small_sample = df.sample(n=100, random_state=42)
small_sample.to_csv(
    "sample_100.csv",
    index=False
)