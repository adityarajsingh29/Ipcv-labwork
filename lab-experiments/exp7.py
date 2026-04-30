import cv2
import numpy as np
 
img = cv2.imread('input_image.jpg')
rows, cols = img.shape[:2]
 
# Translation
M_trans = np.float32([[1, 0, 50], [0, 1, 50]])
translated = cv2.warpAffine(img, M_trans, (cols, rows))
 
# Scaling
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
 
# Rotation (45 degrees about center)
M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv2.warpAffine(img, M_rot, (cols, rows))
 
cv2.imshow('Translated', translated)
cv2.imshow('Scaled', scaled)
cv2.imshow('Rotated', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
