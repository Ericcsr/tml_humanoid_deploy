import numpy as np
import torch
import time
import redis
import pickle
from threading import Lock
import pybullet as pb
from scipy.spatial.transform import Rotation
from tml_humanoid_deploy.utils.math_utils import heading_zup

import unitree_sdk2py  # type: ignore
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # type: ignore
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_  # type: ignore
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG  # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG  # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo  # type: ignore
from unitree_sdk2py.utils.crc import CRC  # type: ignore


from tml_humanoid_deploy.utils.robot_utils import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode, RemoteController, KeyMap
from tml_humanoid_deploy.envs.base_env import BaseRobotEnv

class UnitreeRobot(BaseRobotEnv):
    def __init__(self, 
                 net=None, 
                 config=None):
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.target_dof_pos = np.zeros(29, dtype = np.float32)

        self.kp = np.array(config["joint_stiffness"], dtype=np.float32)
        self.kd = np.array(config["joint_damping"], dtype=np.float32)

        self.counter = 0
        self.control_dt = config.get("control_dt", 0.02)
        self.config = config
        assert net is not None
        ChannelFactoryInitialize(0, net)

        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.remote_controller = RemoteController()
        self.control_lock = Lock()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.low_state_subscriber_ = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_subscriber_.Init(self.LowStateHgHandler, 10)
        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        self.terminated = False

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def _get_imu_quat(self):
        if not hasattr(self, "init_heading"):
            self.init_heading = heading_zup(np.array(self.low_state.imu_state.quaternion)[[1,2,3,0]])
            self.init_heading_quat = Rotation.from_euler("z", self.init_heading).as_quat()
        current_quat = np.array(self.low_state.imu_state.quaternion)[[1,2,3,0]]
        current_quat = Rotation.from_quat(current_quat)
        current_quat = (Rotation.from_quat(self.init_heading_quat).inv() * current_quat).as_quat()
        return current_quat

    def _get_omega(self):
        return np.array(self.low_state.imu_state.gyroscope)

    def pd_control(self, target_q, kp= None, kd = None):
        final_kp = self.kp if kp is None else kp
        final_kd = self.kd if kd is None else kd
        
        assert len(target_q) == 29 
        assert len(final_kp) == 29 
        assert len(final_kd) == 29

        for i in range(len(target_q)):
            self.low_cmd.motor_cmd[i].q = target_q[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = final_kp[i]
            self.low_cmd.motor_cmd[i].kd = final_kd[i]
            self.low_cmd.motor_cmd[i].tau = 0
        
        self.send_cmd(self.low_cmd)
    
    def damping_state(self):
        print("Entering damping state")
        self.control_lock.acquire()
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        self.terminated = True
        self.control_lock.release() # Never release lock as this is final command

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while not self.remote_controller.is_start_pressed():
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def single_zero_torque(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)

    def set_robot_state(self, q, kp=None, kd=None): # q 27 dof
        total_time = 2
        num_step = int(total_time/self.control_dt)
        
        init_q = np.zeros(29, dtype=np.float32)
        dof_idx = np.arange(29)
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        for i in dof_idx:
            init_q[i] = self.low_state.motor_state[i].q
    
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha, kp=kp, kd=kd)
            time.sleep(self.control_dt)
    
    def maintain_state(self, q, kp=None, kd=None):
        print("Maintaining state, wait for L1 + A signal...")
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        while not self.remote_controller.is_begin_control_pressed():
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            self.pd_control(target_q, kp=kp, kd=kd)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def get_robot_state(self):
        # self.qj[:] = [state.q for state in self.low_state.motor_state]
        # self.dqj[:] = [state.dq for state in self.low_state.motor_state]
        for i in range(29):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
            # r = Rotation.from_euler("xyz", [0, 0, -self.qj[12]])
            # root_quat = (Rotation.from_quat(self.head_quat) * r).as_quat()
        imu_quat = self._get_imu_quat()
        omega = self._get_omega()
        return self.qj, self.dqj, imu_quat, omega

    def step_robot(self, action):
        self.control_lock.acquire()
        if self.remote_controller.is_stop_pressed():
            self.damping_state()
            self.terminated = True
        if self.terminated:
            print("Exiting...")
            exit(-1)    
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = action
        self.pd_control(target_q)
        # time.sleep(self.control_dt)
        self.control_lock.release()
    