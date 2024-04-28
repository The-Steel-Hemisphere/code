import pyzed.sl as sl
import cv2
import math
import time

zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Get camera information (ZED serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: {0}".format(zed_serial))

# show camera in a window
runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()

orange_lower_left = (36, 50, 50)
orange_upper_left = (70, 255, 255)

DEBUG = True

start = time.time()
i = 0
while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        i += 1
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        left_hsv = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2HSV)

        left_mask = cv2.inRange(left_hsv, orange_lower_left, orange_upper_left)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_OPEN, kernel)
        if DEBUG:
            left_orange_pixels = cv2.bitwise_and(image.get_data(), image.get_data(), mask=left_mask)
        
        left_center = cv2.findNonZero(left_mask)
        left_center = left_center.mean(axis=0).astype(int)


        if DEBUG and left_center is not None:
            # get the center of each orange area, with a radius of 10 pixels
            cv2.circle(left_orange_pixels, tuple(left_center[0]), 10, (0, 255, 0), -1)

        print(left_center)

        x = round(left_center[0][0])
        y = round(left_center[0][1])
        err, point_cloud_value = point_cloud.get_value(x, y)

        if math.isfinite(point_cloud_value[2]):
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
            print(f"Distance to Camera at {{{x};{y}}}: {distance/25.4} inches")
        else : 
            print(f"The distance can not be computed at {{{x};{y}}}")

        
        if DEBUG:
            cv2.imshow("ZED", left_orange_pixels)
        key = cv2.waitKey(1)
        if key == ord('q') or i > 1000:
            break



end = time.time()
print("Frame rate: {0}".format(i / (end - start)))

# Close the camera
zed.close()