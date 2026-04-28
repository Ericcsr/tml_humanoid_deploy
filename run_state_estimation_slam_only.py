"""
State estimation from SLAM (Redis head pose and head velocities) plus leg FK.

Optional IMU–SLAM orientation fusion can be enabled via CLI flag.
Root linear velocity is derived from Redis head_lin_vel / head_ang_vel and joint rates.
"""
from argparse import ArgumentParser

import numpy as np
import pickle
from scipy.spatial.transform import Rotation

from utils.robot_utils import Rate
from utils.robot_model import KinematicsModel

parser = ArgumentParser(
    description="Publish root_data from SLAM + kinematics only (see run_state_estimation.py for full fusion)."
)
parser.add_argument(
    "--use_sim",
    action="store_true",
    default=False,
    help="Use sim mocap link name (torso_link) instead of real (mid360_link).",
)
parser.add_argument(
    "--visualize",
    action="store_true",
    default=False,
    help="PyBullet GUI for kinematics debug.",
)
parser.add_argument(
    "--slow_down",
    type=float,
    default=1.0,
    help="Slow down loop by this factor (same as run_state_estimation.py).",
)
parser.add_argument(
    "--use_imu",
    action="store_true",
    default=False,
    help="Fuse SLAM orientation with IMU quaternion using quaternion_filter.",
)
args = parser.parse_args()

kin_model = KinematicsModel(
    mocap_link_name="torso_link" if args.use_sim else "mid360_link",
    use_slam=False,
    use_foot_odo=False,
    visualize=args.visualize,
    use_acc=False,
    use_sim=args.use_sim,
)
redis_client = kin_model.redis_client

base_hz = 200
rate = Rate(base_hz / args.slow_down)
if args.slow_down != 1.0:
    print(f"[run_state_estimation_slam_only] Slow down: {args.slow_down}x", flush=True)

#redis_client.delete("proprio_data")

while True:
    proprio_data = redis_client.get("proprio_data")
    if proprio_data is None:
        rate.sleep()
        continue
    proprio_data = pickle.loads(proprio_data)
    root_pos, root_orn, root_vel = kin_model.update_root_state_slam_only(
        q=proprio_data[:29],
        dq=proprio_data[29:58],
    )
    if args.use_imu:
        if len(proprio_data) < 65:
            raise ValueError(
                "Expected proprio_data to include IMU quaternion at indices [61:65]."
            )
        imu_quat = np.asarray(proprio_data[61:65], dtype=np.float64)
        root_orn_slam = np.asarray(root_orn, dtype=np.float64)
        root_orn_fused = np.asarray(
            kin_model.quaternion_filter.update(imu_quat, root_orn_slam),
            dtype=np.float64,
        )

        # update_root_state_slam_only returns linear velocity in body frame using SLAM root_orn.
        # Re-express it in the fused body frame for consistency with fused orientation.
        root_vel_world = Rotation.from_quat(root_orn_slam).apply(
            np.asarray(root_vel, dtype=np.float64)
        )
        root_vel = Rotation.from_quat(root_orn_fused).inv().apply(root_vel_world).astype(
            np.float32
        )
        root_orn = root_orn_fused.astype(np.float32)

    root_data = np.hstack((root_pos, root_orn, root_vel))
    redis_client.set("root_data", pickle.dumps(root_data))
    rate.sleep()
