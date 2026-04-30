import cv2
 
# Step 1: Acquire the image from file
image = cv2.imread('input_image.jpg')
 
# Step 2: Check if image is successfully loaded
if image is None:
    print("Error: Image not found")
else:
    # Step 3: Display the acquired image
    cv2.imshow('Acquired Image', image)
 
    # Step 4: Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
