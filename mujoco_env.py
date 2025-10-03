import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy
import pickle
import redis
from utils.math_utils import *

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from utils.robot_utils import Rate
from utils.redis_utils import REDIS_IP, REDIS_PORT

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

# class ElasticBand:

#     def __init__(self):
#         self.stiffness = 200
#         self.damping = 100
#         self.point = np.array([0, 0, 3])
#         self.length = 0
#         self.enable = True

#     def Advance(self, x, dx):
#         """
#         Args:
#           δx: desired position - current position
#           dx: current velocity
#         """
#         δx = self.point - x
#         distance = np.linalg.norm(δx)
#         direction = δx / distance
#         v = np.dot(dx, direction)
#         f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
#         return f

#     def MujuocoKeyCallback(self, key):
#         glfw = mujoco.glfw.glfw
#         if key == glfw.KEY_7:
#             self.length += 0.1
#         if key == glfw.KEY_8:
#             self.length -= 0.1
#         if key == glfw.KEY_9:
#             self.enable = not self.enable

class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True

    def Advance(self, pose):
        """
        Args:
          pose: 13D array containing:
               - pose[0:3]: position in world frame
               - pose[3:7]: quaternion [w,x,y,z] in world frame
               - pose[7:10]: linear velocity in world frame
               - pose[10:13]: angular velocity in world frame
        Returns:
          np.ndarray: 6D vector [fx, fy, fz, tx, ty, tz]
        """
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        δx = self.point - pos
        f = self.kp_pos * (δx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # --- Orientation PD control for torque ---
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # reorder to [x,y,z,w] for scipy
        rot = scipy.spatial.transform.Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()  # axis-angle error
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def disable_band(self):
        self.enable = False

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")

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
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        redis_client = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=0)

        elastic_band = ElasticBand()
        band_attached_link = model.body("torso_link").id

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=elastic_band.MujuocoKeyCallback)
        
        ### prepare shared data
        control = shared_np(30, "control", np.float32)
        q = shared_np(29, "q", np.float32)
        dq = shared_np(29, "dq", np.float32)
        omega_w = shared_np(3, "omega", np.float32)
        imu_quat = shared_np(4, "imu_quat", np.float32)
        root_pos = shared_np(3, "root_pos", np.float32)
        root_vel = shared_np(3, "root_vel", np.float32)
        torso_pos = shared_np(3, "torso_pos", np.float32)
        torso_orn = shared_np(4, "torso_orn", np.float32)
        ### prepare other data
        kp = np.array(config['joint_stiffness'], dtype=np.float32)
        kd = np.array(config['joint_damping'], dtype=np.float32)
        torque_limit = np.array(config['torque_limit'], dtype=np.float32)

        model.opt.timestep = config.get("simulation_dt", 0.005)
        rate = Rate(1/model.opt.timestep)  # 200 Hz by default
        ts = time.time()
        while True:
            
            with control_lock:
                if control[0] > 180:
                    print("Damping mode")
                    tau = damping_control(data, kd)
                elif control[0] < -180:
                    print("Zero torque mode")
                    tau = zero_torque_control()
                else:
                    tau = pd_control(control[:29], data, kp, kd)
            data.ctrl[:] = tau.clip(-torque_limit, torque_limit)
            if control[29] > 100:
                elastic_band.disable_band()
            if elastic_band.enable:
                pose = np.concatenate(
                    [
                        data.xpos[band_attached_link],  # link position in world
                        data.xquat[
                            band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()

                data.xfrc_applied[band_attached_link] = elastic_band.Advance(pose)
            else:
                data.xfrc_applied[band_attached_link] = np.zeros(6)
            mujoco.mj_step(model, data)
            with data_lock:
                q[:] = data.qpos[7:36].copy()
                dq[:] = data.qvel[6:35].copy()
                imu_quat[:] = data.qpos[3:7][[1,2,3,0]].copy()
                omega_w[:] = data.qvel[3:6].copy()
                root_pos[:] = data.qpos[:3].copy()
                root_vel[:] = data.qvel[:3].copy()
                torso_pos[:] = data.xpos[model.body("torso_link").id].copy()
                torso_orn[:] = data.xquat[model.body("torso_link").id][[1,2,3,0]].copy()
            # viewer.render()
            now = time.time()
            if now - ts > 0.1:
                redis_client.set("head_pos", pickle.dumps(data.xpos[model.body("torso_link").id].copy()))
                redis_client.set("head_quat", pickle.dumps(data.xquat[model.body("torso_link").id][[1,2,3,0]].copy()))
                ts = now
            viewer.sync()                                                                  
            rate.sleep()
            #print("Sim step fps:", 1/(time.time() - ts))

class MujocoRobot:
    def __init__(
            self, 
            xml_path, 
            config
        ):
        self.xml_path = xml_path
        
        self.control_var = shared_np(30, "control", np.float32)
        self.q_var = shared_np(29, "q", np.float32)
        self.dq_var = shared_np(29, "dq", np.float32)
        self.omega_var = shared_np(3, "omega", np.float32)
        self.imu_quat_var = shared_np(4, "imu_quat", np.float32)
        self.root_pos = shared_np(3, "root_pos", np.float32)
        self.root_vel = shared_np(3, "root_vel", np.float32)
        self.torso_pos = shared_np(3, "torso_pos", np.float32)
        self.torso_orn = shared_np(4, "torso_orn", np.float32)

        self.control_lock = mp.Lock()
        self.data_lock = mp.Lock()
        self.config = config
        self.control_dt = config.get("control_dt", 0.02) # default 50 Hz
        self.process = mp.Process(target=run_simulation, args=(self.control_lock, self.data_lock, self.xml_path, self.config))
        self.process.start()

    def pd_control(self, target_q):
        with self.control_lock:
            self.control_var[:29] = target_q.copy()

    def release_robot(self):
        with self.control_lock:
            self.control_var[29] = 200.0

    def set_robot_state(self, q):
        total_time = 2
        num_step = int(total_time / self.control_dt)
        with self.data_lock:
            init_q = self.q_var.copy()
        
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha)
            time.sleep(self.control_dt)

    def maintain_state(self, q):
        self.pd_control(q)
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