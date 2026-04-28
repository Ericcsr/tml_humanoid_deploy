"""
Publish root_data directly from mocap Redis data.

This runner reads pickled arrays from three Redis keys:
- root_pos (3,)
- root_quat (4,) in xyzw
- root_lin_vel (3,) in body frame

When --visualize is enabled, this script also updates the robot state in PyBullet
using joint positions from proprio_data and root pose from mocap.
"""
from argparse import ArgumentParser
import pickle
from typing import Any

import numpy as np

from utils.robot_utils import Rate
from utils.robot_model import KinematicsModel
from utils.redis_utils import REDIS_IP, REDIS_PORT


def _to_vec(data: Any, size: int, name: str) -> np.ndarray:
    vec = np.asarray(data, dtype=np.float64).reshape(-1)
    if vec.size != size:
        raise ValueError(f"{name} must have size {size}, got {vec.size}")
    return vec


parser = ArgumentParser(
    description="Read root_pos/root_quat/root_lin_vel Redis keys and publish root_data."
)
parser.add_argument(
    "--root_pos_topic",
    type=str,
    default="root_pos",
    help="Redis key that stores pickled root position.",
)
parser.add_argument(
    "--root_quat_topic",
    type=str,
    default="root_quat",
    help="Redis key that stores pickled root quaternion (xyzw).",
)
parser.add_argument(
    "--root_lin_vel_topic",
    type=str,
    default="root_lin_vel",
    help="Redis key that stores pickled root linear velocity.",
)
parser.add_argument(
    "--proprio_topic",
    type=str,
    default="proprio_data",
    help="Redis key for proprio_data (used only for visualization joint state).",
)
parser.add_argument(
    "--root_data_topic",
    type=str,
    default="root_data",
    help="Redis key to publish packed root_data [pos, quat, lin_vel].",
)
parser.add_argument(
    "--visualize",
    action="store_true",
    default=False,
    help="Enable PyBullet visualization of robot state.",
)
parser.add_argument(
    "--use_sim",
    action="store_true",
    default=False,
    help="Use sim mocap link name (torso_link) instead of real (mid360_link).",
)
parser.add_argument(
    "--slow_down",
    type=float,
    default=1.0,
    help="Slow down loop by this factor.",
)
args = parser.parse_args()

kin_model = KinematicsModel(
    redis_ip=REDIS_IP,
    redis_port=REDIS_PORT,
    mocap_link_name="torso_link" if args.use_sim else "mid360_link",
    use_slam=False,
    use_foot_odo=False,
    visualize=args.visualize,
    use_acc=False,
    use_sim=args.use_sim,
)
redis_client = kin_model.redis_client

base_hz = 50
rate = Rate(base_hz / args.slow_down)
if args.slow_down != 1.0:
    print(f"[run_state_estimation_mocap] Slow down: {args.slow_down}x", flush=True)

last_q = np.zeros(29, dtype=np.float32)

while True:
    root_pos_raw = redis_client.get(args.root_pos_topic)
    root_quat_raw = redis_client.get(args.root_quat_topic)
    root_lin_vel_raw = redis_client.get(args.root_lin_vel_topic)

    if root_pos_raw is None or root_quat_raw is None or root_lin_vel_raw is None:
        rate.sleep()
        continue

    try:
        root_pos = _to_vec(pickle.loads(root_pos_raw), 3, "root_pos")
        root_quat = _to_vec(pickle.loads(root_quat_raw), 4, "root_quat")
        root_lin_vel = _to_vec(pickle.loads(root_lin_vel_raw), 3, "root_lin_vel")
    except Exception as exc:
        print(
            "[run_state_estimation_mocap] Failed to parse mocap Redis keys "
            f"({args.root_pos_topic}, {args.root_quat_topic}, {args.root_lin_vel_topic}): {exc}",
            flush=True,
        )
        rate.sleep()
        continue

    root_data = np.hstack((root_pos, root_quat, root_lin_vel))
    redis_client.set(args.root_data_topic, pickle.dumps(root_data))

    if args.visualize:
        proprio_raw = redis_client.get(args.proprio_topic)
        if proprio_raw is not None:
            try:
                proprio_data = pickle.loads(proprio_raw)
                last_q = np.asarray(proprio_data[:29], dtype=np.float32)
            except Exception as exc:
                print(
                    f"[run_state_estimation_mocap] Failed to parse {args.proprio_topic}: {exc}",
                    flush=True,
                )

        root_pose = np.zeros(7, dtype=np.float32)
        root_pose[:3] = np.asarray(root_pos, dtype=np.float32)
        root_pose[3:] = np.asarray(root_quat, dtype=np.float32)
        kin_model.set_robot_state(last_q, root_pose)

    rate.sleep()
