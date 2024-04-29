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
import sys


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
        
        left_center = cv2.findNonZero(left_mask)
        if left_center is  None:
            if started:
                break
            else:
                continue


        left_center = left_center.mean(axis=0).astype(int)

        # get the bonding box of the continous area of orange around the 'left_center'
        get_bounding_box = cv2.boundingRect(left_mask)
        


        x = round(left_center[0][0])
        y = round(left_center[0][1])
        err, point_cloud_value = point_cloud.get_value(x, y)

        if math.isfinite(point_cloud_value[2]):
            xm = (point_cloud_value[0]/25.4)
            ym = -(point_cloud_value[1]/25.4)
            zm = (point_cloud_value[2]/25.4)
            started = True

            pos_data.append([time.time(), xm, ym, zm, i])
        
        if len(pos_data) >= 15:
            time1 = time.time()
            print("Data collected")
            
            break
    else:
        print("Failed to read frame from the camera")
        break

end = time.time()
# Close the camera
zed.close()

new_start = time.time()

pos_data = np.asarray(pos_data)

frame_time = pos_data[:,0]
x = pos_data[:,1]
y = pos_data[:,2]
z = pos_data[:,3]
i = pos_data[:,4]

projectile_x = x
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


(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
next_mean = smoothed_state_means[-1]
next_covar = smoothed_state_covariances[-1]
predicted_error = []
predicted_state_means = []
predicated_state_covariances = []
pitch = []
yaw = []
launcher_offset = (0.00635,-0.1778,0.1524)
predict_count  = -1
degree_speed = 191 # degrees per second
yaw_distance = []
time_to_prediction = []
time_advantage = []
time_per_frame = 1/framerate
yaw_time  = []
time_diff = []
advantage_count = 0
time_to_predict = time.time() - time1
while True:
    next_mean, next_covar = kf1.filter_update(next_mean,next_covar)
    predicted_state_means.append(next_mean)
    predicated_state_covariances.append(next_covar)
    predict_count = predict_count + 1
    #t
    time_diff.append(predict_count * time_per_frame - time_to_predict)
    time_to_prediction.append(predict_count*time_per_frame + time_to_predict)
    
    launcher_distance = tuple(np.subtract(next_mean[0:3],launcher_offset))
    temp_yaw = 90-math.degrees(np.arctan(launcher_distance[2]/launcher_distance[0]))
    if temp_yaw >90:
        temp_yaw = temp_yaw - 180
    yaw.append(temp_yaw)
    yaw_distance.append(abs(90-math.degrees(np.arctan(launcher_distance[2]/launcher_distance[0]))))
    yaw_time.append(abs(yaw[-1]/degree_speed))
    #if predict_count >= 13:
    #    break
    time_advantage.append(time_to_prediction[-1]- yaw_time[-1]-0.05)
    if time_advantage[-1] > 0:
        advantage_count = advantage_count + 1
        if advantage_count >= 7:
            break
    
    pitch.append(abs(math.degrees(np.arctan(launcher_distance[1]/(np.sqrt(np.power(launcher_distance[2],2) + np.power(launcher_distance[0],2)))))))
abs_pitch = np.abs(pitch)
time2 = time.time()
predicted_state_means = np.array(predicted_state_means)
print(yaw)
print(pitch)
#print("Time for kalman filter is " + str(time2-time1) + " seconds")
if abs_pitch[-1] < 90 and abs_pitch[-1] >0 and yaw[-1]>-58.5 and yaw[-1] < 61.5:
    if abs_pitch[-1] < 4:
        move_string = f"move({abs_pitch[-1]},{yaw[-1]-1.5})\r\f"
    else:
        move_string = f"move({abs_pitch[-1]-4},{yaw[-1]-1.5})\r\f"
    launch_delay = 0.01
    
    t.write(move_string.encode('utf-8'))
    fire_string = f"launch({launch_delay})\r\f"
    #time.sleep(time_advantage[-1]-launch_delay)
    t.write(fire_string.encode('utf-8'))
print(time_advantage)