import cv2

# Create a VideoCapture object for the stereo camera
cap = cv2.VideoCapture(2)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Get the width and height of the camera feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region of interest (ROI) for each half of the width
roi_left = (0, 0, width // 2, height)
roi_right = (width // 2, 0, width // 2, height)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from the camera")
        break

    # Split the frame into left and right halves
    left_frame = frame[roi_left[1]:roi_left[1] + roi_left[3], roi_left[0]:roi_left[0] + roi_left[2]]
    right_frame = frame[roi_right[1]:roi_right[1] + roi_right[3], roi_right[0]:roi_right[0] + roi_right[2]]

    # Detect balls in each half of the width
    # get the orange pixels in each frame

    # Convert the frames to the HSV color space
    left_hsv = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color in HSV
    orange_lower_left = (15, 60, 70)
    orange_upper_left = (30, 255, 255)

    orange_lower_right = (15, 60, 70)
    orange_upper_right = (30, 255, 255)

    # Create a mask for the orange color in each frame
    left_mask = cv2.inRange(left_hsv, orange_lower_left, orange_upper_left)
    right_mask = cv2.inRange(right_hsv, orange_lower_right, orange_upper_right)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # remove any noise in the mask
    left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_OPEN, kernel)
    right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_OPEN, kernel)


    # Apply the mask to the frames to get the orange pixels
    left_orange_pixels = cv2.bitwise_and(left_frame, left_frame, mask=left_mask)
    right_orange_pixels = cv2.bitwise_and(right_frame, right_frame, mask=right_mask)

    # get the center of each orange area, with a radius of 10 pixels
    left_center = cv2.findNonZero(left_mask)
    right_center = cv2.findNonZero(right_mask)

    # Draw a circle at the center of the orange area
    if left_center is not None:
        left_center = left_center.mean(axis=0).astype(int)
        cv2.circle(left_orange_pixels, tuple(left_center[0]), 10, (0, 255, 0), -1)

    if right_center is not None:
        right_center = right_center.mean(axis=0).astype(int)
        cv2.circle(right_orange_pixels, tuple(right_center[0]), 10, (0, 255, 0), -1)

    # use the center of each orange area to calculate the depth via stereo vision
    
    cv2.stereoCalibrate()
        

    # Display the frames
    cv2.imshow("Left Frame", left_orange_pixels)
    cv2.imshow("Right Frame", right_orange_pixels)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()