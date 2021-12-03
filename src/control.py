import numpy as np 
import rospy
from vision_2 import JointEstimation2

def calculate_A(alpha, a, theta):
    # since d is always 0 it is not included
    A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
              [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
              [0, np.sin(alpha), np.cos(alpha), 0],
              [0,0,0,1]])
    return A

def calculate_end_effector_position():
    theta_1, theta_2, theta_3, theta_4 = [0, 90, 0, 0]
    alpha_1, alpha_2, alpha_3, alpha_4 = [90, 90, 90, 0]
    a_1, a_2, a_3, a_4 = [4, 0, 3.2, 2.8]

    A_1 = calculate_A(alpha_1, a_1, theta_1)
    A_2 = calculate_A(alpha_2, a_2, theta_2)
    A_3 = calculate_A(alpha_3, a_3, theta_3)
    A_4 = calculate_A(alpha_4, a_4, theta_4)

    T = A_1 @ A_2 @ A_3 @ A_4

    end_effector_position = list(T[:-1, -1])

    return end_effector_position

# Code used to calculate positions for our 10 points 

def calculate_10_positions():
    estimated_positions = []
    errors = []
    joint_angles = [[1, 1, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1], [0.5, 1, 0.5], [0.5, 0.5, 1], [1, 0.5, 0.5], 
            [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 0.5, 1.5]]

    desired_positions = [[4.8, -0.114, 6.666], [5.371, -1.981, 7.428], [3.162, 1.029, 8.533], [4.762, -3.086, 6.59], 
                        [4.533, -4.533, 7.238], [3.467, -1.029, 8.533], [3.2, -0.381, 9.524], [2.895, -2.057, 9.41], 
                        [2.705, 1.448, 9.486], [1.562, 3.162, 6.857]]

    for joint_angle, position in zip(joint_angles, desired_positions):
        estimated_position = calculate_end_effector_position(*joint_angle)
        error = np.sum(np.abs(np.array(position) - np.array(estimated_position))) 
        estimated_positions.append(estimated_position)
        errors.append(error)

    return estimated_positions, errors

#################### 
    
def calculate_jacobian(theta_1, theta_3, theta_4):

    jacobian = np.array([[2.8*(np.cos(theta_4)*np.cos(theta_1)*np.sin(theta_3) - np.sin(theta_4)*np.sin(theta_1))
                        - 4*np.sin(theta_1), 0, 2.8*(np.cos(theta_4) * np.sin(theta_1) * np.cos(theta_3)) + 
                        3.2*np.sin(theta_4)*np.cos(theta_3), 
                        2.8*(-np.sin(theta_4) * np.sin(theta_1) * np.sin(theta_3) + np.cos(theta_4)*np.cos(theta_1)) + 
                        3.2 * np.cos(theta_4) * np.sin(theta_4)],
                        [2.8*(np.sin(theta_1) * np.sin(theta_3) * np.cos(theta_4) + np.cos(theta_1)*np.sin(theta_4)) +
                        3.2 * np.sin(theta_1) * np.sin(theta_3) + 4*np.cos(theta_1), 0, 2.8*(-np.cos(theta_1) * 
                        np.cos(theta_3)*np.cos(theta_4)) - 3.2*np.cos(theta_1)*np.cos(theta_3), 
                        2.8*(np.cos(theta_1)*np.sin(theta_3) * np.sin(theta_4) + np.sin(theta_1)*np.cos(theta_4))], 
                        [0, 0, -2.8*np.sin(theta_3)*np.cos(theta_4) - 3.2*np.sin(theta_3), -2.8*np.cos(theta_3) * 
                        np.sin(theta_4)]])

    return jacobian

# Not working properly
def control_open(image, prev_time):

    je = JointEstimation2()
    # estimate time step
    cur_time = rospy.get_time()
    dt = cur_time - prev_time
    prev_time = cur_time
    q = je.estimate_joint_angles(image) # estimate initial value of joints'
    J_inv = np.linalg.pinv(calculate_jacobian(image))  # calculating the psudeo inverse of Jacobian
    pos = je.detect_red(image)
    # desired trajectory
    pos_d= 0 # trajectory()
    
    error = (pos_d - pos)/dt
    q_d = q + (dt * np.dot(J_inv, error.transpose()))  # desired joint angles to follow the trajectory
    return q_d, cur_time