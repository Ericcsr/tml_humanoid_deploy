#!/usr/bin/env python3
"""
SIM ground-truth root publisher — the --use_sim replacement for superodometry +
run_state_estimation_slam_only.py.

In sim there is no lidar/SLAM, but MuJoCo IS the ground truth. run_controller.py
with --use_odom blocks at startup waiting for Redis ``root_data`` (normally
written by the SLAM estimator). This process supplies that key directly from the
sim's ground-truth pelvis pose, so NO estimator is needed.

It attaches (read-only) to the named SharedMemory blocks that mujoco_env.py's sim
subprocess creates — ``root_pos``(3), ``imu_quat``(4, xyzw), ``root_vel``(3, world)
— and publishes:

    root_data = [root_pos(3) | root_orn(4, xyzw) | root_vel(3, body)]

matching exactly what run_state_estimation_slam_only.py writes and what both
run_controller.py and vla_motion_bridge.py read. root_vel is rotated world->body
(same as MujocoRobot.get_root_state).

``proprio_data`` is NOT written here — run_controller.py --use_odom already writes
it from env.get_robot_state(). This process only owns the GT root.

Cleanup NEVER unlinks the shared memory (the sim owns it); it only detaches.
"""
import argparse
import pickle
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.spatial.transform import Rotation as R
import redis

sys.path.insert(0, __file__.rsplit("/vla_bridge/", 1)[0])
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402

# (name, size) of the sim's ground-truth SharedMemory blocks we read.
SHM_SPEC = {"root_pos": 3, "imu_quat": 4, "root_vel": 3}


def _attach_all(timeout):
    """Attach to every SHM block, retrying until the sim has created them."""
    t0 = time.time()
    while True:
        try:
            shms, views = {}, {}
            for name, size in SHM_SPEC.items():
                shm = SharedMemory(create=False, name=name)
                shms[name] = shm
                views[name] = np.ndarray(size, dtype=np.float32, buffer=shm.buf)
            return shms, views
        except FileNotFoundError:
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"sim SharedMemory {list(SHM_SPEC)} not found after {timeout}s "
                    "(is run_controller.py --use_sim running?)"
                )
            time.sleep(0.2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--hz", type=float, default=200.0, help="Publish rate (matches estimator base_hz).")
    p.add_argument("--attach_timeout", type=float, default=60.0)
    args = p.parse_args()

    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    print("[sim_gt] attaching to sim SharedMemory ...", flush=True)
    shms, v = _attach_all(args.attach_timeout)
    print("[sim_gt] attached. Publishing GT root_data @ "
          f"{args.hz:.0f}Hz (no estimator).", flush=True)

    period = 1.0 / args.hz
    next_tick = time.perf_counter()
    n = 0
    try:
        while True:
            root_pos = np.asarray(v["root_pos"], np.float64).copy()
            root_orn = np.asarray(v["imu_quat"], np.float64).copy()      # xyzw
            root_vel_w = np.asarray(v["root_vel"], np.float64).copy()    # world
            nrm = np.linalg.norm(root_orn)
            if nrm < 1e-6:      # sim not stepping yet
                time.sleep(period)
                continue
            root_orn /= nrm
            root_vel_b = R.from_quat(root_orn).inv().apply(root_vel_w)   # world -> body
            root_data = np.hstack((root_pos, root_orn, root_vel_b)).astype(np.float32)
            rdb.set("root_data", pickle.dumps(root_data))
            n += 1
            next_tick += period
            dt = next_tick - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        for shm in shms.values():
            shm.close()      # detach only; the sim owns/unlinks these.
        print(f"[sim_gt] stopped after {n} root_data updates.", flush=True)


if __name__ == "__main__":
    main()
