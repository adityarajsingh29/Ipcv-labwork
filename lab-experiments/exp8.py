import cv2
import numpy as np
 
img = cv2.imread('input.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
# Define range for green color in HSV
lower = np.array([30, 50, 50])
upper = np.array([90, 255, 255])
 
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(img, img, mask=mask)
 
cv2.imshow("Original", img)
cv2.imshow("Segmented", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
