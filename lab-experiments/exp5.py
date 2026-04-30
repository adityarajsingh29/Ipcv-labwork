import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('input_image.jpg', 0)
 
# Gaussian Noise
gauss = img + np.random.normal(0, 20, img.shape)
gauss = np.clip(gauss, 0, 255).astype(np.uint8)
 
# Salt & Pepper Noise
sp = img.copy()
prob = 0.02
rand = np.random.rand(*img.shape)
sp[rand < prob] = 0
sp[rand > 1 - prob] = 255
 
mean_filt = cv2.blur(gauss, (3, 3))
median_filt = cv2.medianBlur(sp, 3)
 
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(gauss, cmap='gray'); plt.title('Gaussian Noise'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(mean_filt, cmap='gray'); plt.title('Mean Filter'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(median_filt, cmap='gray'); plt.title('Median Filter'); plt.axis('off')
plt.show()
