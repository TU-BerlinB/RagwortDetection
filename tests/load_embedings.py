import numpy as np

data = np.load("outputs/embeddings.npz")

X = data["embeddings"]
y = data["labels"]
paths = data["paths"]