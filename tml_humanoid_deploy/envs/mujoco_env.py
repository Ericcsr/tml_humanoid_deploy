import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy
import pickle
from tml_humanoid_deploy.utils.math_utils import *

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from tml_humanoid_deploy.utils.robot_utils import Rate
from tml_humanoid_deploy.envs.base_env import BaseRobotEnv

G1_LINKS = [
    'pelvis', 
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_track_site',    # idx: 5
    'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_track_site',   # idx: 12
    'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link',
    'torso_link',
    'head_track_site',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'left_hand_track_site',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 
    'right_hand_track_site'
]

ZERO_TORQUE = -np.ones(29, dtype=np.float32) * 200.0
DAMPING = np.ones(29, dtype=np.float32) * 200.0

shm_buffer = []

def shared_np(size, name, dtype=np.float32):
    try:
        shm = SharedMemory(create=True, size=np.prod(size) * np.dtype(dtype).itemsize, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    except FileExistsError:
        print("Shared memory already exists")
        shm = SharedMemory(create=False, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    return arr

def pd_control(target_q, data, kp, kd):
    return (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd

def damping_control(data, kd):
    return -data.qvel[6:35] * kd

def zero_torque_control():
    return np.zeros(29, dtype=np.float32)

def run_simulation(control_lock, data_lock, xml_path, config):
        with open(xml_path, 'r') as f:
            xml = f.read()
        
        print(xml_path)
        model = mujoco.MjModel.from_xml_string(xml) # type: ignore
        data = mujoco.MjData(model)  # type: ignore

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=None)
        
        ### prepare shared data 
        control = shared_np(29, "control", np.float32)
        q = shared_np(29, "q", np.float32)
        dq = shared_np(29, "dq", np.float32)
        omega_w = shared_np(3, "omega", np.float32)
        imu_quat = shared_np(4, "imu_quat", np.float32)
        root_pos = shared_np(3, "root_pos", np.float32)
        root_vel = shared_np(3, "root_vel", np.float32)
        torso_pos = shared_np(3, "torso_pos", np.float32)
        torso_orn = shared_np(4, "torso_orn", np.float32)
        kp_shared = shared_np(29, "kp", np.float32)
        kd_shared = shared_np(29, "kd", np.float32)

        torque_limit = np.array(config['torque_limit'], dtype=np.float32)
        
        model.opt.timestep = config.get("simulation_dt", 0.005)
        rate = Rate(1/model.opt.timestep)  # 200 Hz by default
        
        while True:
            #ts = time.time()
            with control_lock:
                tau = pd_control(control[:29], data, kp_shared, kd_shared)
            data.ctrl[:] = tau.clip(-torque_limit, torque_limit)
            
            mujoco.mj_step(model, data)  # type: ignore
            with data_lock:
                q[:] = data.qpos[7:36].copy()
                dq[:] = data.qvel[6:35].copy()
                imu_quat[:] = data.qpos[3:7][[1,2,3,0]].copy()
                omega_w[:] = data.qvel[3:6].copy()
                root_pos[:] = data.qpos[:3].copy()
                root_vel[:] = data.qvel[:3].copy()
                torso_pos[:] = data.xpos[model.body("torso_link").id].copy()
                torso_orn[:] = data.xquat[model.body("torso_link").id][[1,2,3,0]].copy()
            viewer.sync()                                                                  
            rate.sleep()
            # print("Sim step fps:", 1/(time.time() - ts))

class MujocoRobot(BaseRobotEnv):
    def __init__(
            self, 
            xml_path, 
            config
        ):
        self.xml_path = xml_path
        
        self.control_var = shared_np(29, "control", np.float32)
        self.q_var = shared_np(29, "q", np.float32)
        self.dq_var = shared_np(29, "dq", np.float32)
        self.omega_var = shared_np(3, "omega", np.float32)
        self.imu_quat_var = shared_np(4, "imu_quat", np.float32)
        self.root_pos = shared_np(3, "root_pos", np.float32)
        self.root_vel = shared_np(3, "root_vel", np.float32)
        self.torso_pos = shared_np(3, "torso_pos", np.float32)
        self.torso_orn = shared_np(4, "torso_orn", np.float32)
        self.kp_var = shared_np(29, "kp", np.float32)
        self.kd_var = shared_np(29, "kd", np.float32)
        
        # Initialize with default gains from config
        self.kp_var[:] = np.array(config['joint_stiffness'], dtype=np.float32)
        self.kd_var[:] = np.array(config['joint_damping'], dtype=np.float32)
        
        self.control_lock = mp.Lock()
        self.data_lock = mp.Lock()
        self.config = config
        self.control_dt = config.get("control_dt", 0.02) # default 50 Hz

        self.process = mp.Process(target=run_simulation, args=(self.control_lock, self.data_lock, self.xml_path, self.config))
        self.process.start()

    def pd_control(self, target_q, kp=None, kd=None):
        with self.control_lock:
            self.control_var[:29] = target_q.copy()
            self.kp_var[:] = np.array(kp if kp is not None else self.config['joint_stiffness'], dtype=np.float32)
            self.kd_var[:] = np.array(kd if kd is not None else self.config['joint_damping'], dtype=np.float32)

    def set_robot_state(self, q, kp=None, kd=None):
        total_time = 2
        num_step = int(total_time / self.control_dt)
        with self.data_lock:
            init_q = self.q_var.copy()
        
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha, kp=kp, kd=kd)
            time.sleep(self.control_dt)

    def maintain_state(self, q, kp=None, kd=None):
        self.pd_control(q, kp=kp, kd=kd)
        input("Press Enter to continue...")

    def damping_state(self):
        self.pd_control(DAMPING)
    
    def zero_torque_state(self):
        self.pd_control(ZERO_TORQUE)

    def get_robot_state(self):
        with self.data_lock:
            q = self.q_var.copy()
            dq = self.dq_var.copy()
            imu_quat = self.imu_quat_var.copy()
            omega_w = self.omega_var.copy()
        omega_l = omega_w #Rotation.from_quat(imu_quat).inv().apply(omega_w)
        return q, dq, imu_quat, omega_l
    
    def get_root_state(self):
        with self.data_lock:
            root_pos = self.root_pos.copy()
            root_vel = self.root_vel.copy()
            root_orn = self.imu_quat_var.copy()
        root_vel_l = Rotation.from_quat(root_orn).inv().apply(root_vel)
        return root_pos, root_orn, root_vel_l
    
    def get_anchor_state(self):
        with self.data_lock:
            torso_pos = self.torso_pos.copy()
            torso_orn = self.torso_orn.copy()
        return torso_pos, torso_orn

    def step_robot(self, action):
        self.pd_control(action)

    def close(self):
        self.process.terminate()
        self.process.join()


if __name__ == "__main__":
    import yaml
    with open("sample_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    robot = MujocoRobot("assets/g1/scene_29dof.xml", config["rl_policy"])
    
    time.sleep(1)  # wait for the simulation to start

    robot.zero_torque_state()

    input()

    robot.set_robot_state(np.zeros(29, dtype=np.float32))
    for i in range(100):
        print(robot.get_robot_state())
        time.sleep(0.01)
    input()
    
    robot.damping_state()

    input()
    
    robot.close()