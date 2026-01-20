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

# Check if motion has 'wrist_grasp_label'
has_grasp_label = False
if "wrist_grasp_label" in motion.keys():
    print("Motion has grasp label")
    has_grasp_label = True

motion_length = motion["joint_pos"].shape[0]

fps = 50#float(motion["fps"])

init_root_pos = motion["body_pos_w"][0,0]
init_root_pos[2] = 0  # set initial height to 0
init_root_heading = Rotation.from_quat(heading_quat(motion["body_quat_w"][0,0][[1,2,3,0]]))  # xyzw to wxyz
all_links = [pb.getJointInfo(robot, i)[12].decode("utf-8") for i in range(pb.getNumJoints(robot))]
hand_names = ["left_rubber_hand", "right_rubber_hand"]
hand_ids = [all_links.index(name) for name in hand_names]

while True:
    for i in range(motion_length):
        joint_pos = motion["joint_pos"][i]
        set_joint_angles(robot, joint_pos[ISAAC_TO_MUJOCO])
        rel_root_pos = init_root_heading.inv().apply(motion["body_pos_w"][i,0] - init_root_pos)
        rel_root_orn = (init_root_heading.inv() * Rotation.from_quat(motion["body_quat_w"][i,0][[1,2,3,0]])).as_quat()
        if has_grasp_label: # draw sphere around wrist if grasp label is true
            grasp_label = motion["wrist_grasp_label"][i]
            if grasp_label > 0.5:
                pb.addUserDebugText("O", pb.getLinkState(robot, hand_ids[0])[0], textColorRGB=[1,0,0], textSize=3, lifeTime=2.0/fps)
                pb.addUserDebugText("O", pb.getLinkState(robot, hand_ids[1])[0], textColorRGB=[1,0,0], textSize=3, lifeTime=2.0/fps)
        pb.resetBasePositionAndOrientation(robot, rel_root_pos, rel_root_orn)
        time.sleep(1.0 / fps)
        #breakpoint()
    input("Press Enter to replay the motion...")
