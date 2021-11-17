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

def callback():
    return
# run the code if the node is called
if __name__ == '__main__':
    try:
        vision_1_task2_1_solution()
    except rospy.ROSInterruptException:
        pass

