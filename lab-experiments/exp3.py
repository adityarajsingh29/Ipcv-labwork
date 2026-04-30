import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('input_image.jpg', 0)
 
# Contrast Stretching
min_val, max_val = np.min(img), np.max(img)
cs_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
 
# Histogram Equalization
he_img = cv2.equalizeHist(img)
 
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray')
plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(cs_img, cmap='gray')
plt.title('Contrast Stretching'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(he_img, cmap='gray')
plt.title('Histogram Equalization'); plt.axis('off')
plt.show()
