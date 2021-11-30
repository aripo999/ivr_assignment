import numpy as np 

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