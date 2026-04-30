import cv2
 
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Apply Canny Edge Detector
# Parameters: image, lower threshold, upper threshold
edges = cv2.Canny(gray, 100, 200)
 
cv2.imshow("Original", gray)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
