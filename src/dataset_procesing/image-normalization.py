import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

folder = Path("images_small")

# image path loop
for image_path in folder.iterdir(): 

  image_cv = cv2.imread(image_path)

  if image_cv is None:
    raise FileNotFoundError("Image not found. Check the file path.")

  image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

  # Normalize to [0, 1]
  normalized_image1_cv = cv2.normalize(image_cv_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  #----checking----
  print('on initial image')
  print("min_pixel = " , image_cv_rgb.min()  , "and max_pixel = ", image_cv_rgb.max())

  print('on normalized image 1')
  print("min_pixel = " , normalized_image1_cv.min()  , "and max_pixel = ", normalized_image1_cv.max())

  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.imshow(normalized_image1_cv)#, cmap='gray')
  plt.title("normalized image 1 cv")
  plt.axis('off') 
  plt.tight_layout()
  plt.show()