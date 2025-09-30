import pybullet as pb
import numpy as np
from scipy.spatial.transform import Rotation
import time
from argparse import ArgumentParser
from utils.params import ISAAC_TO_MUJOCO
from utils.math_utils import heading_quat

def set_joint_angles(robot_id, joint_angles):
    j = 0
    for i in range(pb.getNumJoints(robot_id)):
        if pb.getJointInfo(robot_id, i)[2] == pb.JOINT_REVOLUTE:
            pb.resetJointState(robot_id, i, joint_angles[j])
            j += 1


parser = ArgumentParser()
parser.add_argument("--motion", type=str, required=True, help="Path to the motion file.")
args = parser.parse_args()

c = pb.connect(pb.GUI)

robot = pb.loadURDF("assets/g1/g1_29dof_kin_extended.urdf")
frame = pb.loadURDF("assets/frame.urdf")

motion = np.load(args.motion)   

motion_length = motion["joint_pos"].shape[0]

fps = 50#float(motion["fps"])

init_root_pos = motion["body_pos_w"][0,0]
init_root_pos[2] = 0  # set initial height to 0
init_root_heading = Rotation.from_quat(heading_quat(motion["body_quat_w"][0,0][[1,2,3,0]]))  # xyzw to wxyz
frame_id = 13
while True:
    for i in range(motion_length):
        joint_pos = motion["joint_pos"][i]
        set_joint_angles(robot, joint_pos[ISAAC_TO_MUJOCO])
        rel_root_pos = init_root_heading.inv().apply(motion["body_pos_w"][i,0] - init_root_pos)
        rel_root_orn = (init_root_heading.inv() * Rotation.from_quat(motion["body_quat_w"][i,0][[1,2,3,0]])).as_quat()
        rel_frame_pos = init_root_heading.inv().apply(motion["body_pos_w"][i,frame_id] - init_root_pos)
        rel_frame_orn = (init_root_heading.inv() * Rotation.from_quat(motion["body_quat_w"][i,frame_id][[1,2,3,0]])).as_quat()

        pb.resetBasePositionAndOrientation(robot, rel_root_pos, rel_root_orn)
        pb.resetBasePositionAndOrientation(frame, rel_frame_pos, rel_frame_orn)
        time.sleep(1.0 / fps)
    input("Press Enter to replay the motion...")
