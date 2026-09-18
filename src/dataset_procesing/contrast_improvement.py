# Histogram equalization to improve contrast
import cv2
import numpy as np

from pathlib import Path

image_cv = cv2.imread("images_small/1966622080.jpg")

if image_cv is None:
  raise FileNotFoundError("Image not found. Check the file path.")


# Convert from BGR to YCrCb color space
ycrcb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YCrCb)

# Split into channels
y, cr, cb = cv2.split(ycrcb)

# Apply histogram equalization only on the Y channel (luminance)
y_eq = cv2.equalizeHist(y)

# Merge channels back
ycrcb_eq = cv2.merge((y_eq, cr, cb))

# Convert back to BGR color space
image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Convert back to BGR color space
image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# For matplotlib, convert BGR -> RGB
image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
image_eq_rgb = cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)

# Display
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_cv_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_eq_rgb)
plt.title("Equalized")
plt.axis("off")

plt.tight_layout()
plt.show()