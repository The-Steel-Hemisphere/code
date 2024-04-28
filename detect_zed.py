import pyzed.sl as sl
import cv2
import math
import time
import numpy as np

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math
import matplotlib.animation as animation
from sympy import Symbol, solve
from math import*
from pykalman import KalmanFilter
from talker import Talker
import serial

# establish a ctrl+c handler
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    exitProgram()
    sys.exit(0)

t = serial.Serial("/dev/ttyACM0", 115200, timeout=10)
#t.write(b">>>\r\f")
print("written")
time.sleep(1)
calibrate_string = f"calibrate()\r\f"
t.write(calibrate_string.encode('utf-8'))
print("calibrated")


zed = sl.Camera()


video = []
print ("dd")
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)
init_params.camera_fps = 60
#init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(err)
    exit(1)

# Get camera information (ZED serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: {0}".format(zed_serial))

# show camera in a window
runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()


# These parameters are for blue
#orange_lower_left = (110, 50, 50)
#orange_upper_left = (170, 255, 255)



orange_lower_left = (36, 100, 100)
orange_upper_left = (70, 255, 255)

DEBUG = False
start = time.time()
i = 0
pos_data = []

started = False

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        i += 1
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        blurred = cv2.GaussianBlur(image.get_data(), (11, 11), 0)
        left_hsv = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2HSV)

        left_mask = cv2.inRange(left_hsv, orange_lower_left, orange_upper_left)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_OPEN, kernel)
        if DEBUG:
            left_orange_pixels = cv2.bitwise_and(image.get_data(), image.get_data(), mask=left_mask)
        
        left_center = cv2.findNonZero(left_mask)
        if left_center is  None:
            if started:
                break
            else:
                continue


        left_center = left_center.mean(axis=0).astype(int)

        if DEBUG and left_center is not None:
            # get the center of each orange area, with a radius of 10 pixels
            cv2.circle(left_orange_pixels, tuple(left_center[0]), 10, (0, 255, 0), -1)

        # get the bonding box of the continous area of orange around the 'left_center'
        get_bounding_box = cv2.boundingRect(left_mask)
        if DEBUG and get_bounding_box is not None:
            # draw the bounding box
            cv2.rectangle(left_orange_pixels, get_bounding_box, (0, 255, 0), 2)

        


        x = round(left_center[0][0])
        y = round(left_center[0][1])
        err, point_cloud_value = point_cloud.get_value(x, y)

        if math.isfinite(point_cloud_value[2]):
            xm = (point_cloud_value[0]/25.4)
            ym = -(point_cloud_value[1]/25.4)
            zm = (point_cloud_value[2]/25.4)
            #print(f"Distance to Camera at {{{x};{y}}}: xm, ym, zm = {xm}, {ym}, {zm}")
            started = True
            font = cv2.FONT_HERSHEY_SIMPLEX

            # cv2.putText(left_orange_pixels, f"xm, ym, zm = {xm}, {ym}, {zm}", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            pos_data.append([time.time(), xm, ym, zm, i])

        #else : 
            #print(f"The distance can not be computed at {{{x};{y}}}")
        
        # save the image to the vidoe
        # add in text the output of the distance
        


        
        if DEBUG:
            cv2.imshow("ZED", left_orange_pixels)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if len(pos_data) >= 15:
            print("Data collected")
            break


    else:
        print("Failed to read frame from the camera")
        break

# save the data to a csv file
# import csv
# with open(f'position_data{time.time()}.csv', mode='w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['time', 'x', 'y', 'z'])
#     for row in pos_data:
#         writer.writerow(row)

# # save the file
# print(f"Data saved to position_data.csv")


# end = time.time()
# print("Frame rate: {0}".format(i / (end - start)))
end = time.time()
# Close the camera
zed.close()


# def exitProgram():

#     # save all the images to a folder
#     import os
#     os.makedirs('video', exist_ok=True)
#     for i, img in enumerate(video):
#         cv2.imwrite(f'video/{i}.png', img)

#     print(f"Video saved to video folder")   

#     cv2.destroyAllWindows()


# # make a matplot lib 3d plot GIF of the data
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

#import matplotlib.animation as animation
new_start = time.time()
frame_time = [x[0] for x in pos_data]

x = [x[1] for x in pos_data]
y = [x[2] for x in pos_data]
z = [x[3] for x in pos_data]
i = [x[4] for x in pos_data]
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')

# def animate(i):
#     ax.scatter(x[0:i], z[0:i], y[0:i])

# ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=100)


#plt.show()


# exitProgram()

projectile_x = np.asarray(x)
framerate = len(projectile_x)/(frame_time[-1]-frame_time[0])
#print(projectile_x)
projectile_x = projectile_x[10:]
projectile_x = projectile_x /25.4
projectile_y = np.asarray(y)
projectile_y = projectile_y /25.4 
projectile_y = projectile_y[10:]
projectile_z = np.asarray(z)
#print(projectile_z)
projectile_z = projectile_z /25.4
projectile_z = projectile_z[10:]
#print(len(projectile_x))

#framerate = 11/(projectile_df.iloc[10]['time']-projectile_df.iloc[0]['time'])
#framerate = 53
print("Framerate is " + str(framerate))
initial_x = []
initial_y = []
initial_z = []
initial_velocity = (projectile_y[1] - projectile_y[0]) * framerate
#initial_x_velocity = (projectile_x[1]-projectile_x[0]) * framerate
#initial_z_velocity = (projectile_z[1]-projectile_z[0]) * framerate

initial_x_velocity = np.diff(projectile_x).mean() * framerate
initial_z_velocity = np.diff(projectile_z).mean() * framerate
#initial_x_velocity = ((projectile_x[1]-projectile_x[0]) + (projectile_x[2]-projectile_x[1]) + (projectile_x[3]-projectile_x[2]))/3 * framerate
#initial_z_velocity=  ((projectile_z[1]-projectile_z[0]) + (projectile_z[2]-projectile_z[1]) + (projectile_z[3]-projectile_z[2]))/3 * framerate
print(f"Initial x velocity is {initial_x_velocity}")
print(f"Initial y velocity is {initial_velocity}")
print(f"Initial z velocity is {initial_z_velocity}")
dT = 1 / framerate
g = 9.80665
initial_state = np.asarray([projectile_x[0],projectile_y[0],projectile_z[0],initial_x_velocity,initial_velocity,initial_z_velocity,0,-1*g,0])
transition_matrix = np.asarray(
    [
        [1., 0., 0., dT, 0., 0., 0.5*dT*dT, 0., 0.], # x pos
        [0., 1., 0., 0., dT, 0., 0., 0.5*dT*dT, 0.], # y pos
        [0., 0., 1., 0., 0., dT, 0., 0., 0.5*dT*dT], # z pos
        [0., 0., 0., 1., 0., 0., dT, 0., 0.], # x velocity
        [0., 0., 0., 0., 1., 0., 0., dT, 0.], # y velocity
        [0., 0., 0., 0., 0., 1., 0., 0., dT], # z velocity
        [0., 0., 0., 0., 0., 0., 1., 0., 0.], # x accel
        [0., 0., 0., 0., 0., 0., 0., 1., 0.], # y accel
        [0., 0., 0., 0., 0., 0., 0., 0., 1.] # z accel
    ]
)
observation_matrix = np.asarray(
    [
        [1, 0, 0, 0, 0, 0,0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ]
)





kf1 = KalmanFilter(transition_matrices = transition_matrix,
                observation_matrices = observation_matrix,
                initial_state_mean = initial_state)
measurements = []

for values in zip(projectile_x,projectile_y,projectile_z):
    measurements.append(values)

time1 = time.time()
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
next_mean = smoothed_state_means[-1]
next_covar = smoothed_state_covariances[-1]
predicted_error = []
predicted_state_means = []
predicated_state_covariances = []
pitch = []
yaw = []
launcher_offset = (0.00635,-0.1778,0.1524)
predict_count  = 1
degree_speed = 191 # degrees per second
yaw_distance = []
time_to_prediction = []
time_advantage = []
time_per_frame = 1/framerate
yaw_time  = []
time_to_predict = time.time() - time1
advantage_count = 0
print(time_to_predict)
while True:
    next_mean, next_covar = kf1.filter_update(next_mean,next_covar)
    predicted_state_means.append(next_mean)
    predicated_state_covariances.append(next_covar)
    predict_count = predict_count + 1
    time_to_prediction.append(predict_count*time_per_frame + time_to_predict)
    
    launcher_distance = tuple(np.subtract(next_mean[0:3],launcher_offset))
    temp_yaw = 90-math.degrees(np.arctan(launcher_distance[2]/launcher_distance[0]))
    if temp_yaw >90:
        temp_yaw = temp_yaw - 180
    yaw.append(temp_yaw)
    yaw_distance.append(abs(90-math.degrees(np.arctan(launcher_distance[2]/launcher_distance[0]))))
    yaw_time.append(abs(yaw[-1]/degree_speed))
    time_advantage.append(time_to_prediction[-1]- yaw_time[-1] -0.01)
    if time_advantage[-1] > 0:
        advantage_count = advantage_count + 1
        if advantage_count >= 8:
            break
    
    pitch.append(abs(math.degrees(np.arctan(launcher_distance[1]/(np.sqrt(np.power(launcher_distance[2],2) + np.power(launcher_distance[0],2)))))))
abs_pitch = np.abs(pitch)
time2 = time.time()
predicted_state_means = np.array(predicted_state_means)
print(yaw)
print(pitch)
#print("Time for kalman filter is " + str(time2-time1) + " seconds")
if abs_pitch[-1] < 90 and abs_pitch[-1] >5 and yaw[-1]>-59.5 and yaw[-1] < 60.5:
    move_string = f"move({abs_pitch[-1]-5},{yaw[-1]-0.5})\r\f"
    launch_delay = 0.01
    
    t.write(move_string.encode('utf-8'))
    fire_string = f"launch({launch_delay})\r\f"
    #time.sleep(time_advantage[-1]-launch_delay)
    t.write(fire_string.encode('utf-8'))
print(time_advantage)