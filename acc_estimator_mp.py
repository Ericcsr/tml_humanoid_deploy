import numpy as np
import time
import redis
import pickle
import multiprocessing as mp
from real_env import UnitreeRobot

# Use Shared Memory for non-blocking data exchange
# Size calculation: 29 (ddq) + 29 (tau) + 3 (lin_acc) + 3 (imu_alpha) = 64 floats
SHARED_ARRAY_SIZE = 64 

class AccelerationEstimator:
    def __init__(self, num_joints=29, alpha_filter=0.2):
        self.num_joints = num_joints
        self.alpha = alpha_filter
        self.prev_dq = np.zeros(num_joints, dtype=np.float32)
        self.prev_omega = np.zeros(3, dtype=np.float32)
        self.prev_time = None
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
        if dt > 1e-5: # Avoid div by zero
            raw_ddq = (current_dq - self.prev_dq) / dt
            self.ddq[:] = self.alpha * raw_ddq + (1.0 - self.alpha) * self.ddq
            raw_alpha = (current_omega - self.prev_omega) / dt
            self.imu_alpha[:] = self.alpha * raw_alpha + (1.0 - self.alpha) * self.imu_alpha
            
            self.prev_dq[:] = current_dq
            self.prev_omega[:] = current_omega
            self.prev_time = curr_time
        return self.ddq, self.imu_alpha

def redis_worker(shared_data, stop_event, redis_freq=50):
    """Dedicated process for Redis I/O."""
    r = redis.Redis(host='localhost', port=6379, db=0)
    loop_dt = 1.0 / redis_freq
    
    while not stop_event.is_set():
        start_time = time.perf_counter()
        
        # Read from shared memory safely
        with shared_data.get_lock():
            # Convert shared array to numpy for easy slicing
            data = np.frombuffer(shared_data.get_obj(), dtype=np.float32)
            ddq = data[0:29]
            tau = data[29:58]
            lin_acc = data[58:61]
            imu_alpha = data[61:64]

        try:
            # Batch these updates in a pipeline if performance drops
            r.set("root_a", pickle.dumps(np.hstack([lin_acc, imu_alpha])))
            r.set("ddq", pickle.dumps(ddq))
            r.set("tau", pickle.dumps(tau))
        except Exception as e:
            print(f"Redis Error: {e}")

        # Maintain 50Hz
        elapsed = time.perf_counter() - start_time
        time.sleep(max(0, loop_dt - elapsed))

def main():
    # 1. Initialization
    config = {"control_dt": 0.002, "joint_stiffness": [20]*29, "joint_damping": [0.5]*29}
    robot = UnitreeRobot(net="enp5s0", config=config)
    estimator = AccelerationEstimator(num_joints=29, alpha_filter=0.15)
    
    # 2. Setup Multiprocessing
    # 'd' for double-precision, 'f' for float. Using 'f' to match robot SDK.
    shared_buffer = mp.Array('f', SHARED_ARRAY_SIZE)
    stop_event = mp.Event()
    
    p_redis = mp.Process(target=redis_worker, args=(shared_buffer, stop_event))
    p_redis.start()

    print("Estimation Loop: Running at 500Hz (Maximum Effort)")
    print("Redis Process: Decoupled and running at 50Hz")

    try:
        while not robot.terminated:
            loop_start = time.perf_counter()
            
            # --- HIGH PRIORITY ESTIMATION ---
            q, dq, quat, omega = robot.get_robot_state()
            tau = np.array([robot.low_state.motor_state[i].tau_est for i in range(29)], dtype=np.float32)
            lin_acc = -np.array(robot.low_state.imu_state.accelerometer, dtype=np.float32)
            
            ddq, imu_alpha = estimator.update(dq, omega)

            # --- UPDATE SHARED MEMORY ---
            # Lock is brief: just a memory copy, no I/O
            with shared_buffer.get_lock():
                buf = np.frombuffer(shared_buffer.get_obj(), dtype=np.float32)
                buf[0:29] = ddq
                buf[29:58] = tau
                buf[58:61] = lin_acc
                buf[61:64] = imu_alpha
            
            # Precise timing
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, config["control_dt"] - elapsed))
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        p_redis.join()
        robot.release_robot()

if __name__ == "__main__":
    main()