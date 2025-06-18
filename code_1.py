import cv2
import time
import numpy as np

# Preparation for writing the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Reading from the webcam
cap = cv2.VideoCapture(0)

# Allow the system to sleep for 3 seconds before the webcam starts
time.sleep(3)

count = 0
background = 0

# Dynamically capture the background continuously
def capture_background(cap):
    ret, background = cap.read()
    if not ret:
        return None
    return np.flip(background, axis=1)

# Read every frame from the webcam, until the camera is open
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)

    # Continuously update the background on every frame
    background = capture_background(cap)

    # Resize the background to match the current frame size
    background = cv2.resize(background, (img.shape[1], img.shape[0]))

    # Convert the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])  # Lower bound of black
    upper_black = np.array([180, 255, 50])  # Upper bound of black (low brightness)

    # Generate the mask for black color
    mask1 = cv2.inRange(hsv, lower_black, upper_black)

    # Open and Dilate the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverted mask to segment out the black color from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segment the black color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    # Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    # Ensure both images have the same size before blending
    if res1.shape != res2.shape:
        res2 = cv2.resize(res2, (res1.shape[1], res1.shape[0]))

    # Convert res2 to uint8 type if it's not already
    if res2.dtype != np.uint8:
        res2 = np.uint8(res2)

    # Ensure both images are in the same number of channels
    if len(res1.shape) == 2:
        res1 = cv2.cvtColor(res1, cv2.COLOR_GRAY2BGR)
    if len(res2.shape) == 2:
        res2 = cv2.cvtColor(res2, cv2.COLOR_GRAY2BGR)

    # Generate the final output and write it
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(finalOutput)
    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
