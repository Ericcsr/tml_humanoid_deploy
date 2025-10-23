from argparse import ArgumentParser
import numpy as np
from scipy.spatial.transform import Rotation

from utils.robot_utils import Rate
from utils.robot_model import KinematicsModel
import redis
import pickle

parser = ArgumentParser()
parser.add_argument("--use_sim", action="store_true", default=False,
                    help="Whether to use simulation or real robot")
parser.add_argument("--use_slam", action="store_true", default=False,
                    help="Whether to use SLAM for root state estimation")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Whether to visualize the kinematics model")
args = parser.parse_args()



kin_model = KinematicsModel(mocap_link_name="torso_link" if args.use_sim else "mid360_link", 
                            use_slam=args.use_slam, visualize=args.visualize)
redis_client = kin_model.redis_client

rate = Rate(50)  # 50 Hz

redis_client.set("proprio_data", None)

while True:
    proprio_data = pickle.loads(redis_client.get("proprio_data"))
    if proprio_data is None:
        rate.sleep()
        continue
    #breakpoint()
    root_pos, root_orn, root_vel = kin_model.update_root_state(
        q=proprio_data[:29], 
        dq=proprio_data[29:58], 
        omega=proprio_data[58:61],
        imu_quat=proprio_data[61:65]
    )
    root_data = np.hstack((root_pos, root_orn, root_vel))
    redis_client.set("root_data", pickle.dumps(root_data))
    rate.sleep()
