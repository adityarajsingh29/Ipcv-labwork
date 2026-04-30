import cv2
import numpy as np
 
img = cv2.imread('input.jpg', 0)
flat = img.flatten()
 
# Run Length Encoding
def rle_encode(data):
    encoding = []
    prev = data[0]
    count = 1
    for i in data[1:]:
        if i == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = i
            count = 1
    encoding.append((prev, count))
    return encoding
 
encoded = rle_encode(flat)
 
print("Original size:", len(flat))
print("Encoded size :", len(encoded))
print(f"Compression ratio: {len(flat) / len(encoded):.2f}")
