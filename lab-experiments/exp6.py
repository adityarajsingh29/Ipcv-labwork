from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('input_image.jpg', 0)
 
# Add periodic noise
noise = 20 * np.sin(np.linspace(0, 50, img.shape[1]))
noise = noise.reshape(1, -1)
noisy = img + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
 
# Apply Wiener filter
wiener_img = wiener(noisy)
 
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(noisy, cmap='gray'); plt.title('Noisy Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(wiener_img, cmap='gray'); plt.title('Wiener Filtered'); plt.axis('off')
plt.show()
