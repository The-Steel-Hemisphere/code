import cv2
import time

# Open the default camera
cap = cv2.VideoCapture(2)

# set the frame rate to 60 FPS
cap.set(cv2.CAP_PROP_FPS, 120)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video capture.")
    exit()

# Get the initial time
start_time = time.time()

# Initialize frame count
frame_count = 0

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading video frame.")
        break

    # Display the frame
    cv2.imshow("Frame", frame)

    # Increment frame count
    frame_count += 1

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate the frame rate
frame_rate = frame_count / elapsed_time

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

# Print the frame rate
print("Frame rate: {:.2f} FPS".format(frame_rate))