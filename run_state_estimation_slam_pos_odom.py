"""
State estimation: SLAM for root position, IMU+SLAM orientation fusion, foot odometry for velocity.

Unlike run_state_estimation.py (full pos_filter / odom–SLAM blend), root xy and z follow SLAM only
(after slam_pos_to_world). Foot contact kinematics are used only for linear velocity, not to shift
published root position.
"""
from argparse import ArgumentParser

import numpy as np
import pickle

from utils.robot_utils import Rate
from utils.robot_model import KinematicsModel

parser = ArgumentParser(
    description="root_data: SLAM position + fused orientation + foot odom velocity."
)
parser.add_argument(
    "--use_sim",
    action="store_true",
    default=False,
    help="QuaternionCollaborativeFilterSimple use_sim (pass with MuJoCo).",
)
parser.add_argument(
    "--use_acc",
    action="store_true",
    default=False,
    help="Use ForceTorqueFootOdometer (requires ddq/root_a/tau on Redis).",
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
    help="Slow down loop by this factor.",
)
args = parser.parse_args()

kin_model = KinematicsModel(
    mocap_link_name="mid360_link" if not args.use_sim else "torso_link",
    use_slam=False,
    use_foot_odo=True,
    visualize=args.visualize,
    use_acc=args.use_acc,
    use_sim=args.use_sim,
)
redis_client = kin_model.redis_client

base_hz = 50
rate = Rate(base_hz / args.slow_down)
if args.slow_down != 1.0:
    print(f"[run_state_estimation_slam_pos_odom] Slow down: {args.slow_down}x", flush=True)

redis_client.delete("proprio_data")

while True:
    proprio_data = redis_client.get("proprio_data")
    if proprio_data is None:
        rate.sleep()
        continue
    proprio_data = pickle.loads(proprio_data)
    if not args.use_acc:
        root_pos, root_orn, root_vel = kin_model.update_root_state_slam_pos_fused_orn_foot_vel(
            q=proprio_data[:29],
            imu_quat=proprio_data[61:65],
            dq=proprio_data[29:58],
            omega=proprio_data[58:61],
        )
    else:
        ddq = pickle.loads(redis_client.get("ddq"))
        root_a = pickle.loads(redis_client.get("root_a"))
        tau = pickle.loads(redis_client.get("tau"))
        root_pos, root_orn, root_vel = kin_model.update_root_state_slam_pos_fused_orn_foot_vel(
            q=proprio_data[:29],
            imu_quat=proprio_data[61:65],
            dq=proprio_data[29:58],
            omega=proprio_data[58:61],
            ddq=ddq,
            root_a=root_a,
            tau=tau,
        )
    root_data = np.hstack((root_pos, root_orn, root_vel))
    redis_client.set("root_data", pickle.dumps(root_data))
    rate.sleep()
