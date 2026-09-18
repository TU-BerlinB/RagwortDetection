#Test for thershold in blury images and clear images in data set from a small sample

import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def blur_score(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

for folder in ["blurry", "sharp"]:
    print(f"\n{folder.upper()}")

    scores = []

    for img_path in Path(folder).glob("*"):
        score = blur_score(img_path)
        scores.append(score)

    print(f"Min: {min(scores):.2f}")
    print(f"Max: {max(scores):.2f}")
    print(f"Avg: {sum(scores)/len(scores):.2f}")

blurry_scores = []
sharp_scores = []

def blur_score(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Process blurry images
for img in Path("blurry").glob("*"):
    score = blur_score(img)

    if score is not None:
        blurry_scores.append(score)

# Process sharp images
for img in Path("sharp").glob("*"):
    score = blur_score(img)

    if score is not None:
        sharp_scores.append(score)

plt.hist(blurry_scores, alpha=0.6, label="Blurry")
plt.hist(sharp_scores, alpha=0.6, label="Sharp")

plt.xlabel("Laplacian Variance")
plt.ylabel("Count")
plt.legend()

plt.show()