import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# Read the image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
 
# Function for sampling
def sampling(img, factor):
    return img[::factor, ::factor]
 
# Function for quantization
def quantization(img, levels):
    img_norm = img / 255.0
    img_quant = np.floor(img_norm * levels) / levels
    return (img_quant * 255).astype(np.uint8)
 
# Sampling factors and quantization levels
sampling_factors = [1, 2, 4]
quant_levels = [256, 64, 16]
 
# Display sampling results
plt.figure(figsize=(10, 4))
for i, f in enumerate(sampling_factors):
    sampled_img = sampling(image, f)
    plt.subplot(1, 3, i+1)
    plt.imshow(sampled_img, cmap='gray')
    plt.title(f'Sampling Factor = {f}')
    plt.axis('off')
plt.show()
 
# Display quantization results
plt.figure(figsize=(10, 4))
for i, q in enumerate(quant_levels):
    quant_img = quantization(image, q)
    plt.subplot(1, 3, i+1)
    plt.imshow(quant_img, cmap='gray')
    plt.title(f'Quantization Levels = {q}')
    plt.axis('off')
plt.show()
