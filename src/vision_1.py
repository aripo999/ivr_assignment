#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class JointEstimation1:

    def __init__(self):
        # initialize the node
        rospy.init_node('vision_1_task2_1_solution', anonymous=True)
        rate = rospy.Rate(50)  # 50hz

        # initialize a publisher for each joint
        self.joint2_est_pub = rospy.Publisher("/robot/joint2_estimation", Float64, queue_size=10)
        self.joint3_est_pub = rospy.Publisher("/robot/joint3_estimation", Float64, queue_size=10)
        self.joint4_est_pub = rospy.Publisher("/robot/joint4_estimation", Float64, queue_size=10)

        # initialize a subscriber for each camera
        camera_1_sub = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        camera_2_sub = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)

        self.cv_image_x = np.array([], dtype=np.uint8)
        self.cv_image_y = np.array([], dtype=np.uint8)

        self.pixel2meter_factor = None

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        
        while not rospy.is_shutdown():
            rate.sleep()

    # Detecting the centre of the red circle
    def detect_red(self, image):
        # Isolate the red colour in the image as a binary image
        mask = cv2.inRange(image, (0, 0, 100), (10, 10, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        except ZeroDivisionError:  # return None if joint is not visible
            return None

    # Detecting the centre of the green circle
    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (20, 255, 10))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        except ZeroDivisionError:
            return None

    # Detecting the centre of the blue circle
    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 10, 10))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        except ZeroDivisionError:
            return None

    # Detecting the centre of the yellow circle
    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        except ZeroDivisionError:
            return None

    # Calculate the conversion from pixel to meter
    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        circle1Pos = self.detect_green(image)
        circle2Pos = self.detect_yellow(image)
        # find the distance between two circles
        dist = np.sum((circle1Pos - circle2Pos)**2)
        return 4 / np.sqrt(dist)


    # Calculate the relevant joint angles from the image
    def estimate_joint_angles(self,image_x, image_y):
        # only calculate conversion factor if it hasn't been calculated before
        if self.pixel2meter_factor is None:
            self.pixel2meter_factor = self.pixel2meter(image_x)
        a = self.pixel2meter_factor

        # Obtain the centre of each coloured blob
        green_x = self.detect_green(image_x)
        green_y = self.detect_green(image_y)
        green1 = 0 if green_y is None else a * green_y[0]
        green2 = 0 if green_x is None else a * green_x[0]
        green3 = a * green_y[1] if green_x is None else a * green_x[1]

        yellow_x = self.detect_yellow(image_x)
        yellow_y = self.detect_yellow(image_y)
        yellow1 = green1 if yellow_y is None else a * yellow_y[0]
        yellow2 = green2 if yellow_x is None else a * yellow_x[0]
        yellow3 = a * yellow_y[1] if yellow_x is None else a * yellow_x[1]

        blue_x = self.detect_blue(image_x)
        blue_y = self.detect_blue(image_y)
        blue1 = yellow1 if blue_y is None else a * blue_y[0]  # assume that blue joint is hidden behind yellow joint
        blue2 = yellow2 if blue_x is None else a * blue_x[0]  # if it's not visible from one of the cameras
        blue3 = a * blue_y[1] if blue_x is None else a * blue_x[1]

        red_x = self.detect_red(image_x)
        red_y = self.detect_red(image_y)
        red1 = blue1 if red_y is None else a * red_y[0]
        red2 = blue2 if red_x is None else a * red_x[0]
        red3 = a * red_y[1] if red_x is None else a * red_x[1]

        # Solve using trigonometry
        ja2 = - np.arctan2(yellow1 - blue1, yellow3 - blue3)
        ja3 = np.arctan2(yellow2 - blue2, yellow3 - blue3)
        ja4 = - np.arctan2(blue1 - red1, blue3 - red3) - ja2

        return np.array([ja2, ja3, ja4])

    # Find the joint angles and publish them
    def process_and_publish(self):
        if self.cv_image_x.size != 0 and self.cv_image_y.size != 0:
            angles = self.estimate_joint_angles(self.cv_image_x, self.cv_image_y)

            joint2_est = Float64()
            joint2_est.data = angles[0]
            self.joint2_est_pub.publish(joint2_est)

            joint3_est = Float64()
            joint3_est.data = angles[1]
            self.joint3_est_pub.publish(joint3_est)

            joint4_est = Float64()
            joint4_est.data = angles[2]
            self.joint4_est_pub.publish(joint4_est)

    def callback1(self, data):
        # Receive the image from camera 1
        try:
          self.cv_image_x = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        self.process_and_publish()

    def callback2(self, data):
        # Receive the image from camera 2
        try:
          self.cv_image_y = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        self.process_and_publish()

# run the code if the node is called
if __name__ == '__main__':
    try:
        est = JointEstimation1()
    except rospy.ROSInterruptException:
        pass

