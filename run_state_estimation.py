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
parser.add_argument("--use_acc", action="store_true", default=False)
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Whether to visualize the kinematics model")
args = parser.parse_args()


class LowPassFilter:
    """
    First‑order exponential low‑pass filter on arrays of arbitrary shape.

    y[k] = y[k-1] + alpha * (x[k] - y[k-1])

    Attributes:
        alpha (float): smoothing factor in [0,1]. 
            Smaller α → stronger smoothing (lower cutoff frequency).
        state (np.ndarray): last filtered value (same shape as inputs).
        initialized (bool): True once the first sample sets the state.
    """

    def __init__(self, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.state = None
        self.initialized = False

    def reset(self):
        """Clear the filter state; the next input will reinitialize it."""
        self.state = None
        self.initialized = False

    def filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the low‑pass filter to input x.

        Args:
            x (array‑like): new sample (any shape). Will be cast to np.ndarray.

        Returns:
            np.ndarray: filtered output, same shape as x.
        """
        x = np.array(x, copy=False)
        if not self.initialized:
            # On first call, just set state = x
            self.state = x.astype(float)
            self.initialized = True
        else:
            # y = y_prev + α (x - y_prev)
            self.state += self.alpha * (x - self.state)
        return self.state

kin_model = KinematicsModel(mocap_link_name="torso_link" if args.use_sim else "mid360_link", 
                            use_slam=args.use_slam, visualize=args.visualize, use_acc=args.use_acc)
redis_client = kin_model.redis_client

rate = Rate(50)  # 50 Hz

redis_client.delete("proprio_data")
root_pos_low_pass = LowPassFilter(alpha=0.5)
while True:
    proprio_data = redis_client.get("proprio_data")
    if proprio_data is None:
        rate.sleep()
        continue
    proprio_data = pickle.loads(proprio_data)
    if not args.use_acc:
        root_pos, root_orn, root_vel = kin_model.update_root_state(
            q=proprio_data[:29], 
            dq=proprio_data[29:58], 
            omega=proprio_data[58:61],
            imu_quat=proprio_data[61:65]
        )
    else:
        ddq = pickle.loads(redis_client.get("ddq"))
        root_a = pickle.loads(redis_client.get("root_a"))
        tau = pickle.loads(redis_client.get("tau"))
        print(root_a)
        root_pos, root_orn, root_vel = kin_model.update_root_state(
            q=proprio_data[:29], 
            dq=proprio_data[29:58], 
            omega=proprio_data[58:61],
            imu_quat=proprio_data[61:65],
            ddq = ddq,
            root_a = root_a,
            tau = tau
        )
    root_pos = root_pos_low_pass.filter(root_pos)
    root_data = np.hstack((root_pos, root_orn, root_vel))
    redis_client.set("root_data", pickle.dumps(root_data))
    rate.sleep()
