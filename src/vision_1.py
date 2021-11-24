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
        # Defines publisher and subscriber
        # initialize the node named
        rospy.init_node('vision_1_task2_1_solution', anonymous=True)
        rate = rospy.Rate(50)  # 50hz
        # initialize a publisher for each joint
        joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        joint4_est_pub = rospy.Publisher("/robot/joint4_estimation", Float64, queue_size=10)

        camera_1_sub = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        camera_2_sub = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.cv_image_x = np.array([])
        self.cv_image_y = np.array([])

        t0 = rospy.get_time()
        while not rospy.is_shutdown():
            cur_time = np.array([rospy.get_time()]) - t0
            #y_d = float(6 + np.absolute(1.5* np.sin(cur_time * np.pi/100)))
            joint2_value = np.pi/2 * np.sin(np.pi / 15 * cur_time)
            joint3_value = np.pi/2 * np.sin(np.pi / 20 * cur_time)
            joint4_value = np.pi/2 * np.sin(np.pi / 18 * cur_time)

            joint2 = Float64()
            joint2.data = 0

            joint3 = Float64()
            joint3.data = 0

            joint4 = Float64()
            joint4.data = joint4_value

            joint2_pub.publish(joint2)
            joint3_pub.publish(joint3)
            joint4_pub.publish(joint4)
            if self.cv_image_x.size != 0 and self.cv_image_y.size != 0:
                cv2.imshow('window1', self.cv_image_x)
                cv2.waitKey(1)
                cv2.imshow('window2', self.cv_image_y)
                cv2.waitKey(1)
                joint4_est = Float64()
                joint4_est.data = self.detect_joint_angles(self.cv_image_x, self.cv_image_y)[2]
                joint4_est_pub.publish(joint4_est)

            rate.sleep()

    # In this method you can focus on detecting the centre of the red circle
    def detect_red(self, image):
        # Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0, 100, 0), (20, 255, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        except ZeroDivisionError:
            return None


    # Detecting the centre of the green circle
    def detect_green(self, image):
        mask = cv2.inRange(image, (41, 100, 0), (70, 255, 255))
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
        mask = cv2.inRange(image, (110, 100, 0), (130, 255, 255))
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
        mask = cv2.inRange(image, (21, 100, 0), (40, 255, 255))
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

    def angle(self, x, y, z):
        right = x - y
        left = z - y
        cosine_angle = np.dot(right, left) / (np.linalg.norm(right) * np.linalg.norm(left))
        factor = -1 if right[0] * left[1] - right[1] * left[0] > 0 else 1
        return factor * np.arccos(cosine_angle)

    # Calculate the relevant joint angles from the image
    def detect_joint_angles(self,image_1, image_2):
        a = self.pixel2meter(image_1)

        image_x = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
        image_y = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV)

        # Obtain the centre of each coloured blob
        green_x = self.detect_green(image_x)
        green_y = self.detect_green(image_y)
        center1 = 0 if green_y is None else a * green_y[0]
        center2 = 0 if green_x is None else a * green_x[0]
        center3 = a * green_x[1] if green_y is None else a * green_y[1]
        center = np.array([center1, center2, center3])

        yellow_x = self.detect_yellow(image_x)
        yellow_y = self.detect_yellow(image_y)
        circle1Pos1 = 0 if yellow_y is None else a * yellow_y[0]
        circle1Pos2 = 0 if yellow_x is None else a * yellow_x[0]
        circle1Pos3 = a * yellow_x[1] if yellow_y is None else a * yellow_y[1]
        circle1Pos = np.array([circle1Pos1, circle1Pos2, circle1Pos3])

        blue_x = self.detect_blue(image_x)
        blue_y = self.detect_blue(image_y)
        circle2Pos1 = 0 if blue_y is None else a * blue_y[0]
        circle2Pos2 = 0 if blue_x is None else a * blue_x[0]
        circle2Pos3 = a * blue_x[1] if blue_y is None else a * blue_y[1]
        circle2Pos = np.array([circle2Pos1, circle2Pos2, circle2Pos3])

        red_x = self.detect_red(image_x)
        red_y = self.detect_red(image_y)
        circle3Pos1 = 0 if red_y is None else a * red_y[0]
        circle3Pos2 = 0 if red_x is None else a * red_x[0]
        circle3Pos3 = a * red_x[1] if red_y is None else a * red_y[1]
        circle3Pos = np.array([circle3Pos1, circle3Pos2, circle3Pos3])

        print(center)
        print(circle1Pos)
        print(circle2Pos)
        print(circle3Pos)

        # Solve using trigonometry
        rad = self.angle(2 * circle1Pos - center, circle2Pos, circle3Pos)
        print(np.rad2deg(rad))
        ja1 = np.arctan2(center[0]- circle1Pos[0], center[1] - circle1Pos[1])
        ja2 = np.arctan2(circle1Pos[0]-circle2Pos[0], circle1Pos[1]-circle2Pos[1]) - ja1
        ja3 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja2 - ja1
        return np.array([ja1, ja2, rad])

    def callback1(self, data):
        # Recieve the image
        try:
          self.cv_image_x = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)


    def callback2(self, data):
        # Recieve the image
        try:
          self.cv_image_y = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)


# run the code if the node is called
if __name__ == '__main__':
    try:
        est = JointEstimation1()
    except rospy.ROSInterruptException:
        pass

