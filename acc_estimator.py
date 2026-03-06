import numpy as np
import time
import redis
import pickle
from threading import Lock
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from utils.robot_utils import create_damping_cmd, init_cmd_hg, MotorMode, RemoteController, KeyMap
from real_env import UnitreeRobot

class AccelerationEstimator:
    def __init__(self, num_joints=29, alpha_filter=0.2):
        self.num_joints = num_joints
        self.alpha = alpha_filter
        
        # Pre-allocate buffers for speed
        self.prev_dq = np.zeros(num_joints, dtype=np.float32)
        self.prev_omega = np.zeros(3, dtype=np.float32)
        self.prev_time = None
        
        # Outputs
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.imu_alpha = np.zeros(3, dtype=np.float32)

    def update(self, current_dq, current_omega):
        curr_time = time.perf_counter()
        
        if self.prev_time is None:
            self.prev_time = curr_time
            self.prev_dq[:] = current_dq
            self.prev_omega[:] = current_omega
            return self.ddq, self.imu_alpha

        dt = curr_time - self.prev_time
        if dt > 0:
            # Joint acceleration finite diff + EMA filter
            raw_ddq = (current_dq - self.prev_dq) / dt
            self.ddq[:] = self.alpha * raw_ddq + (1.0 - self.alpha) * self.ddq
            
            # IMU Angular acceleration (alpha) finite diff + EMA filter
            raw_alpha = (current_omega - self.prev_omega) / dt
            self.imu_alpha[:] = self.alpha * raw_alpha + (1.0 - self.alpha) * self.imu_alpha
            
            self.prev_dq[:] = current_dq
            self.prev_omega[:] = current_omega
            self.prev_time = curr_time
            
        return self.ddq, self.imu_alpha

def main(verbose=False):
    # 1. Config & Network
    config = {"control_dt": 0.002, "joint_stiffness": [20]*29, "joint_damping": [0.5]*29}
    network_interface = "enx6c1ff706b392"
    
    # 2. Hardware/Redis Setup
    robot = UnitreeRobot(net=network_interface, config=config)
    estimator = AccelerationEstimator(num_joints=29, alpha_filter=0.15)
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Frequency Management
    loop_dt = config["control_dt"] # 0.002 -> 500Hz
    redis_freq = 50 
    broadcast_every = int((1/redis_freq) / loop_dt) # Every 10 steps
    
    counter = 0
    if verbose:
        print(f"Starting loop at 500Hz. Redis broadcast at {redis_freq}Hz.")

    try:
        while not robot.terminated:
            start_loop_time = time.perf_counter()
            
            # Get raw state from robot
            q, dq, quat, omega = robot.get_robot_state()
            
            # Extract torques and linear acceleration directly from SDK state
            # Efficiently pull motor torques
            tau = np.array([robot.low_state.motor_state[i].tau_est for i in range(29)], dtype=np.float32)
            lin_acc = -np.array(robot.low_state.imu_state.accelerometer, dtype=np.float32)
            
            # Estimate accelerations (500Hz)
            ddq, imu_alpha = estimator.update(dq, omega)

            # 3. Redis Broadcast (50Hz)
            if counter % broadcast_every == 0:
                # payload = {
                #     "joint_acc": ddq,
                #     "joint_tau": tau,
                #     "root_lin_acc": lin_acc,
                #     "root_ang_acc": imu_alpha,
                #     "ts": start_loop_time
                # }
                r.set("root_a", pickle.dumps(np.hstack([lin_acc, imu_alpha])))
                r.set("ddq", pickle.dumps(ddq))
                r.set("tau", pickle.dumps(tau))
            counter += 1
            
            # Precise sleep to maintain 500Hz
            elapsed = time.perf_counter() - start_loop_time
            sleep_time = loop_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        if verbose: print("\nShutting down...")
    finally:
        robot.release_robot()

if __name__ == "__main__":
    # Set verbose=True only for debugging
    main(verbose=True)