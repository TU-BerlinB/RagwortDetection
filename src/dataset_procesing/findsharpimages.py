import cv2
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

THRESHOLD = 100

input_folder = Path("images")
output_folder = Path("sharp_images")

output_folder.mkdir(exist_ok=True)

def blur_score(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_image(img_path):
    score = blur_score(img_path)

    if score is None:
        return "error"

    if score >= THRESHOLD:
        shutil.copy2(
            img_path,
            output_folder / img_path.name
        )
        return "kept"

    return "rejected"

files = list(input_folder.glob("*"))

kept = 0
rejected = 0

with ThreadPoolExecutor(max_workers=16) as executor:

    results = executor.map(process_image, files)

    for result in results:

        if result == "kept":
            kept += 1

        elif result == "rejected":
            rejected += 1

print("\nFinished!")
print(f"Kept: {kept}")
print(f"Rejected: {rejected}")