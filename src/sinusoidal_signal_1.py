#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image

def sinusoidal_signal():

    # initialize the node
    rospy.init_node('sinusoidal_signal_1', anonymous=True)
    rate = rospy.Rate(50)  # 50hz

    # initialize a publisher for each joint
    joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    t0 = rospy.get_time()
    while not rospy.is_shutdown():
        cur_time = np.array([rospy.get_time()]) - t0
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

# run the code if the node is called
if __name__ == '__main__':
    try:
        sinusoidal_signal()
    except rospy.ROSInterruptException:
        pass