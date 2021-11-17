#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64, Image

def vision_1_task2_1_solution():

    # Defines publisher and subscriber
    # initialize the node named
    rospy.init_node('vision_1_task2_1_solution', anonymous=True)
    rate = rospy.Rate(50)  # 50hz
    # initialize a publisher for each joint
    joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    camera_1_sub = rospy.Subscriber("/camera1/robot/image_raw", Image, callback)
    camera_2_sub = rospy.Subscriber("/camera2/robot/image_raw", Image, callback)

    t0 = rospy.get_time()
    while not rospy.is_shutdown():
        cur_time = np.array([rospy.get_time()]) - t0
        #y_d = float(6 + np.absolute(1.5* np.sin(cur_time * np.pi/100)))
        joint2_value = np.pi/2 * np.sin(np.pi / 15 * cur_time)
        joint3_value = np.pi/2 * np.sin(np.pi / 20 * cur_time)
        joint4_value = np.pi/2 * np.sin(np.pi / 18 * cur_time)
        
        joint2 = Float64()
        joint2.data = joint2_value

        joint3 = Float64()
        joint3.data = joint3_value

        joint4 = Float64()
        joint4.data = joint4_value

        joint2_pub.publish(joint2)
        joint3_pub.publish(joint3)
        joint4_pub.publish(joint4)

        rate.sleep()

# In this method you can focus on detecting the centre of the red circle
def detect_red(self,image):
    # Isolate the blue colour in the image as a binary image
    mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
    # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # Obtain the moments of the binary image
    M = cv2.moments(mask)
    # Calculate pixel coordinates for the centre of the blob
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


# Detecting the centre of the green circle
def detect_green(self,image):
    mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


# Detecting the centre of the blue circle
def detect_blue(self,image):
    mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

# Detecting the centre of the yellow circle
def detect_yellow(self,image):
    mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


# Calculate the conversion from pixel to meter
def pixel2meter(self,image):
    # Obtain the centre of each coloured blob
    circle1Pos = self.detect_blue(image)
    circle2Pos = self.detect_green(image)
    # find the distance between two circles
    dist = np.sum((circle1Pos - circle2Pos)**2)
    return 3 / np.sqrt(dist)


# Calculate the relevant joint angles from the image
def detect_joint_angles(self,image):
    a = self.pixel2meter(image)
    # Obtain the centre of each coloured blob 
    center = a * self.detect_yellow(image)
    circle1Pos = a * self.detect_blue(image) 
    circle2Pos = a * self.detect_green(image) 
    circle3Pos = a * self.detect_red(image)
    # Solve using trigonometry
    ja1 = np.arctan2(center[0]- circle1Pos[0], center[1] - circle1Pos[1])
    ja2 = np.arctan2(circle1Pos[0]-circle2Pos[0], circle1Pos[1]-circle2Pos[1]) - ja1
    ja3 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja2 - ja1
    return np.array([ja1, ja2, ja3])
def callback():
    return
# run the code if the node is called
if __name__ == '__main__':
    try:
        vision_1_task2_1_solution()
    except rospy.ROSInterruptException:
        pass

