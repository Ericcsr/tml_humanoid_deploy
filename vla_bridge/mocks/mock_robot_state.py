#!/usr/bin/env python3
"""
MOCK robot-state publisher — stands in for (superodometry + SLAM estimator +
run_controller's proprio feedback), which cannot run on this machine.

Plays a reference clip open-loop at a fixed rate and publishes:
  * root_data   = [root_pos(3) | root_orn(4, xyzw) | root_vel(3)]   (like
    run_state_estimation_slam_only.py writes)
  * proprio_data= [q(29, MuJoCo) | dq(29) | omega(3) | imu_quat(4, xyzw)]  (like
    run_controller.py writes)

The clip is a ``*_bm_norm``-format .npz (joint_pos (T,29) ISAAC, body_pos_w
(T,30,3), body_quat_w (T,30,4) wxyz). This gives the bridge a coherent, moving
world root + proprio so the whole obs->chunk->channels loop can be exercised
locally. NOT a physics sim — pure playback.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import redis

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _DEPLOY_ROOT)
from utils.params import ISAAC_TO_MUJOCO  # noqa: E402
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402

NUM_JOINTS = 29


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True, help="Path to a *_bm_norm .npz clip.")
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--no_loop", dest="loop", action="store_false")
    args = p.parse_args()

    d = np.load(args.clip)
    jp_isaac = d["joint_pos"].astype(np.float32)            # (T,29) ISAAC
    jv_isaac = (d["joint_vel"].astype(np.float32) if "joint_vel" in d.files
                else np.zeros_like(jp_isaac))
    root_pos = d["body_pos_w"][:, 0, :].astype(np.float32)   # (T,3)
    root_quat_wxyz = d["body_quat_w"][:, 0, :].astype(np.float32)  # (T,4) wxyz
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    T = jp_isaac.shape[0]
    fps = float(d["fps"][0]) if "fps" in d.files else 50.0

    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    print(f"[mock_robot] playing {os.path.basename(args.clip)} T={T} @ {args.hz}Hz "
          f"(clip fps={fps}) loop={args.loop}", flush=True)

    # World linear velocity by finite difference (for root_data vel field, body frame).
    from scipy.spatial.transform import Rotation as R
    vel_w = np.zeros((T, 3), np.float32)
    if T >= 2:
        vel_w[:-1] = (root_pos[1:] - root_pos[:-1]) * fps
        vel_w[-1] = vel_w[-2]

    period = 1.0 / args.hz
    i = 0
    next_tick = time.perf_counter()
    while True:
        q_mj = jp_isaac[i][ISAAC_TO_MUJOCO]   # ISAAC -> MuJoCo (mujoco = isaac[ITM])
        dq_mj = jv_isaac[i][ISAAC_TO_MUJOCO]
        vel_b = R.from_quat(root_quat_xyzw[i]).inv().apply(vel_w[i]).astype(np.float32)
        root_data = np.hstack((root_pos[i], root_quat_xyzw[i], vel_b)).astype(np.float32)
        # proprio: q, dq (MuJoCo), omega(0), imu_quat = root orn xyzw.
        proprio = np.hstack((q_mj, dq_mj, np.zeros(3, np.float32), root_quat_xyzw[i])).astype(np.float32)
        rdb.set("root_data", pickle.dumps(root_data))
        rdb.set("proprio_data", pickle.dumps(proprio))

        i += 1
        if i >= T:
            if not args.loop:
                break
            i = 0
        next_tick += period
        dt = next_tick - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            next_tick = time.perf_counter()
    print("[mock_robot] done.", flush=True)


if __name__ == "__main__":
    main()
