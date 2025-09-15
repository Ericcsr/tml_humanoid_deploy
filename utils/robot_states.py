import numpy as np

class G1RobotState:
    def __init__(self):
        ## State from robot proprioception
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self.omega = np.zeros(3)
        self.imu_quat = np.array([1, 0, 0, 0]) # robot wxyz

        ## States from mocap or odometer
        self.root_pos = np.zeros(3)
        self.root_orn = np.array([0, 0, 0, 1]) # mocap xyzw
        self.root_vel = np.zeros(3)

        self.anchor_pos = np.zeros(3)
        self.anchor_orn = np.array([0, 0, 0, 1]) # mocap xyzw


        ## last action
        self.last_action = np.zeros(29)
        