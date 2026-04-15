import os
import pickle
from typing import Optional

import numpy as np
import pybullet as pb
import redis
from scipy.spatial.transform import Rotation

from utils.redis_utils import REDIS_IP, REDIS_PORT
from utils.params import REVOLUTE_JOINTS, ZERO, FOOT_SPHERE_NAMES
from utils.state_estimation import (
    FootOdometer,
    ForceTorqueFootOdometer,
    QuaternionCollaborativeFilterSimple,
    RootPoseFilterSimple,
)

script_path = os.path.realpath(__file__)
current_file_directory = os.path.dirname(script_path)

def get_root_pose_from_link(robot_id, link_id, joint_angles, target_link_pos, target_link_orn):
    """
    Given current joint angles and a target link's desired pose,
    compute the root frame pose.
    Parameters:
    robot_id (int): PyBullet ID of the robot.
    link_id (int): Index of the target link.
    joint_indices (list[int]): Joint indices to set.
    joint_angles (list[float]): Corresponding joint angles.
    target_link_pose (tuple): Desired pose of the link (position, quaternion).
    Returns:
    root_pose (tuple): Computed pose of the robot's root frame (position, quaternion).
    """
    # Reset joints to specified angles
    # pb.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
    current_root_pos, current_root_orn = pb.getBasePositionAndOrientation(robot_id)
    current_root_orn = Rotation.from_quat(current_root_orn)
    current_root_pos = np.array(current_root_pos)
    # set_joint_angles(robot_id, REVOLUTE_JOINTS, joint_angles)
    pb.resetJointStatesMultiDof(robot_id, REVOLUTE_JOINTS, targetValues=joint_angles.reshape(-1, 1))
    # Get the current pose of the link
    state = pb.getLinkState(robot_id, link_id, computeForwardKinematics=True)
    current_link_pos, current_link_orn = np.array(state[4]), np.array(state[5])
    current_link_pos = current_root_orn.inv().apply(current_link_pos - current_root_pos)
    current_link_orn = (current_root_orn.inv() * Rotation.from_quat(current_link_orn)).as_quat()
    # Convert poses to transformation matrices
    T_link_world = np.eye(4)
    T_link_world[:3, :3] = Rotation.from_quat(current_link_orn).as_matrix()
    T_link_world[:3, 3] = current_link_pos
    T_target_link = np.eye(4)
    T_target_link[:3, :3] = Rotation.from_quat(target_link_orn).as_matrix()
    T_target_link[:3, 3] = target_link_pos
    # Compute inverse transform to find root pose
    T_root_world = T_target_link @ np.linalg.inv(T_link_world)
    root_pos = T_root_world[:3, 3]
    root_orn = Rotation.from_matrix(T_root_world[:3, :3]).as_quat()
    # pb.resetBasePositionAndOrientation(robot_id, root_pos, root_orn)
    return root_pos, root_orn

class KinematicsModel:
    def __init__(
        self,
        redis_ip = REDIS_IP,
        redis_port = REDIS_PORT,
        visualize=False,
        ref_robot=False,
        mocap_link_name="mid360_link",
        use_slam=True,
        use_foot_odo=True,
        use_acc=False,
        use_sim=False,
    ):
        pb.connect(pb.GUI if visualize else pb.DIRECT)
        self.redis_client = redis.Redis(redis_ip, port=redis_port, db=0)
        self.robot = pb.loadURDF(f"{current_file_directory}/../assets/g1/g1_29dof_kin_extended.urdf")
        self.link_names = [
            pb.getJointInfo(self.robot, i)[12].decode() for i in range(pb.getNumJoints(self.robot))
        ]
        j = 0
        for i in range(pb.getNumJoints(self.robot)):
            joint_info = pb.getJointInfo(self.robot, i)
            if joint_info[2] == pb.JOINT_REVOLUTE:
                REVOLUTE_JOINTS[j] = i
                j += 1
        if ref_robot:
            self.ref_robot = pb.loadURDF(
                f"{current_file_directory}/../assets/g1/g1_29dof_kin_extended.urdf"
            )
            for link_id in range(-1, pb.getNumJoints(self.ref_robot)):
                pb.changeVisualShape(self.ref_robot, link_id, rgbaColor=[0, 1, 0, 1])

        self.track_sites = [
            pb.loadURDF(f"{current_file_directory}/../assets/frame.urdf") for _ in range(3)
        ]
        self.eef_id = [
            self.link_names.index(name) for name in ["left_rubber_hand", "right_rubber_hand"]
        ]
        pb.setGravity(0.0, 0.0, -9.81)
        self.head_id = self.link_names.index("head_link")
        self.waist_jid = [12, 13, 14]
        self.left_arm_jid = [15, 16, 17, 18, 19, 20, 21]
        self.right_arm_jid = [22, 23, 24, 25, 26, 27, 28]
        self.mocap_link_id = self.link_names.index(mocap_link_name)
        self.root_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.quaternion_filter = QuaternionCollaborativeFilterSimple(use_sim=use_sim)
        self.q = np.zeros(29)
        self.use_acc = use_acc
        if not use_acc:
            self.foot_odo = FootOdometer(self.robot, pb_kin=self, foot_link_names=FOOT_SPHERE_NAMES)
        else:
            self.foot_odo = ForceTorqueFootOdometer(self.robot, pb_kin=self, foot_link_names=FOOT_SPHERE_NAMES)
        self.pos_filter = RootPoseFilterSimple(alpha=0.8)
        self.t = 0
        self.root_pos = np.zeros(3, dtype=np.float32)
        self.root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.use_slam = use_slam
        self.use_foot_odo = use_foot_odo

    def set_robot_state(self, q, root_pose, dq=None, omega=None):
        if dq is not None:
            pb.resetJointStatesMultiDof(
                self.robot,
                REVOLUTE_JOINTS,
                targetValues=q.reshape(-1, 1),
                targetVelocities=dq.reshape(-1, 1),
            )
        else:
            pb.resetJointStatesMultiDof(self.robot, REVOLUTE_JOINTS, targetValues=q.reshape(-1, 1))
        if omega is not None:
            pb.resetBaseVelocity(
                self.robot, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=omega.tolist()
            )
        pb.resetBasePositionAndOrientation(
            self.robot,
            root_pose[:3], #+ Rotation.from_quat(root_pose[3:]).apply(np.array([0.0, 0.0, -0.02])),
            root_pose[3:],
        )

    def set_ref_robot_state(self, q, root_pose):
        pb.resetJointStatesMultiDof(self.ref_robot, REVOLUTE_JOINTS, targetValues=q.reshape(-1, 1))
        pb.resetBasePositionAndOrientation(
            self.ref_robot,
            root_pose[:3] + Rotation.from_quat(root_pose[3:]).apply(np.array([0.0, 0.0, -0.02])),
            root_pose[3:],
        )

    def update_root_state(
        self, q, imu_quat=None, dq=None, omega=None, head_pos=None, head_quat=None, ddq=None, root_a=None, tau=None
    ):
        self.q = q
        if self.use_slam:
            if head_pos is None or head_quat is None:
                self.head_pos = pickle.loads(self.redis_client.get("head_pos"))  # type: ignore
                self.head_quat = pickle.loads(self.redis_client.get("head_quat"))  # type: ignore
            else:
                self.head_pos = head_pos
                self.head_quat = head_quat
            root_pos_slam, root_quat_slam = get_root_pose_from_link(
                self.robot, self.mocap_link_id, self.q, self.head_pos, self.head_quat
            )  # use full kinematic chain


        # fused_quat = self.quaternion_filter.process_imu(imu_quat)
        if self.use_slam:
            self.root_quat = self.quaternion_filter.update(imu_quat, root_quat_slam)
            root_pos_slam = self.quaternion_filter.slam_pos_to_world(root_pos_slam)
            if not self.use_foot_odo:
                pb.resetBasePositionAndOrientation(self.robot, root_pos_slam, self.root_quat)
                self.root_pose = np.concatenate((root_pos_slam, self.root_quat))
                return self.root_pose
        else:
            #self.root_quat = self.quaternion_filter.update(imu_quat, self.root_quat)
            self.root_quat = imu_quat # directly use imu quaternion
        if dq is not None and omega is not None:
            if not self.use_acc:
                root_vel, z = self.foot_odo.estimate_velocity(q, dq, self.root_quat, omega)
            else:
                root_vel, z = self.foot_odo.estimate_velocity(q, dq, self.root_quat, omega, root_a, ddq, tau)
            if self.use_slam:
                # root_pos_slam[2] = root_pos_slam[2] / 2 + z / 2
                # self.root_pos[2] = z
                # self.pos_filter.updateSlam(self.root_pos, timestamp=self.t)
                self.pos_filter.updateSlam(root_pos_slam)
            else:
                self.root_pos[2] = z
                # self.pos_filter.updateSlam(self.root_pos, timestamp=self.t)
                self.pos_filter.updateSlam(self.root_pos)
                root_pos_slam = self.root_pos
            # self.pos_filter.updateOdo(root_vel, timestamp=self.t)
            self.pos_filter.updateOdo(root_vel, z)
            self.t += 0.02
            # self.root_pos, _, _ = self.pos_filter.get_state()
            self.root_pos = self.pos_filter.get_state()
            # self.root_pos[:2] = self.root_pos[:2] * 0.8 + root_pos_slam[:2] * 0.2
            root_pos = self.root_pos
        else:
            root_pos = root_pos_slam
            raise ValueError("dq and omega must be provided for velocity estimation")
        pb.resetBasePositionAndOrientation(self.robot, root_pos, self.root_quat)
        self.root_pose = np.concatenate((root_pos, self.root_quat))
        # Foot odom returns v in world frame; express in root body: v_b = R_wb^{-1} v_w
        root_vel = Rotation.from_quat(self.root_quat).inv().apply(root_vel)
        #return self.root_pose[:3], self.root_pose[3:], root_vel
        return root_pos_slam, self.root_quat, root_vel

    # For visualize and debug slam only
    def update_root_state_slam(
            self, q, imu_quat=None, dq=None, omega=None, head_pos=None, head_quat=None
    ):
        self.q = q
        self.head_pos = pickle.loads(self.redis_client.get("head_pos"))  # type: ignore
        self.head_quat = pickle.loads(self.redis_client.get("head_quat"))
        root_pos_slam, root_quat_slam = get_root_pose_from_link(
            self.robot, self.mocap_link_id, self.q, self.head_pos, self.head_quat
        )  # use full kinematic chain
        pb.resetBasePositionAndOrientation(self.robot, root_pos_slam, root_quat_slam)
        return root_pos_slam, root_quat_slam, np.zeros(3)

    def update_root_state_slam_only(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        head_lin_vel: Optional[np.ndarray] = None,
        head_ang_vel: Optional[np.ndarray] = None,
    ):
        """
        Root pose from SLAM (Redis head_pos / head_quat) and leg FK; root linear velocity from
        Redis head_lin_vel / head_ang_vel (world frame, lidar/head frame) and joint rates.

        v_root_w = v_head_w - ω_root_w × r - v_head_from_joints_w,
        ω_root_w = ω_head_w - ω_head_from_joints_w, with joint-induced velocities from PyBullet
        at zero base twist (returned root_vel is linear in body frame, same as foot odom).
        """
        self.q = q
        if head_pos is None or head_quat is None:
            self.head_pos = pickle.loads(self.redis_client.get("head_pos"))  # type: ignore
            self.head_quat = pickle.loads(self.redis_client.get("head_quat"))  # type: ignore
        else:
            self.head_pos = head_pos
            self.head_quat = head_quat
        if head_lin_vel is None:
            head_lin_vel = pickle.loads(self.redis_client.get("head_lin_vel"))  # type: ignore
        if head_ang_vel is None:
            head_ang_vel = pickle.loads(self.redis_client.get("head_ang_vel"))  # type: ignore
        head_lin_vel = np.asarray(head_lin_vel, dtype=np.float64).reshape(3)
        head_ang_vel = np.asarray(head_ang_vel, dtype=np.float64).reshape(3)

        root_pos, root_quat = get_root_pose_from_link(
            self.robot, self.mocap_link_id, self.q, self.head_pos, self.head_quat
        )
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_quat = np.asarray(root_quat, dtype=np.float64)
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = root_pos.astype(np.float32)
        pose[3:] = root_quat.astype(np.float32)
        self.set_robot_state(q, pose, dq, omega=np.zeros(3, dtype=np.float64))
        ls = pb.getLinkState(
            self.robot,
            self.mocap_link_id,
            computeForwardKinematics=True,
            computeLinkVelocity=True,
        )
        p_mocap = np.asarray(ls[4], dtype=np.float64)
        v_head_joints = np.asarray(ls[6], dtype=np.float64)
        w_head_joints = np.asarray(ls[7], dtype=np.float64)

        w_root_w = head_ang_vel - w_head_joints
        r = p_mocap - root_pos
        v_root_w = head_lin_vel - np.cross(w_root_w, r) - v_head_joints
        root_vel = Rotation.from_quat(root_quat).inv().apply(v_root_w)  # world → root body

        pb.resetBasePositionAndOrientation(self.robot, root_pos, root_quat)
        self.root_quat = root_quat.astype(np.float32)
        self.root_pose = np.concatenate((root_pos, root_quat))
        return root_pos, root_quat, root_vel.astype(np.float32)

    def update_root_state_slam_pos_fused_orn_foot_vel(
        self,
        q: np.ndarray,
        imu_quat: np.ndarray,
        dq: np.ndarray,
        omega: np.ndarray,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        ddq=None,
        root_a=None,
        tau=None,
    ):
        """
        Hybrid estimator: root position entirely from SLAM FK + slam_pos_to_world (no foot-odo
        position filter); orientation from IMU–SLAM fusion (QuaternionCollaborativeFilterSimple);
        linear root velocity from foot odometry (same world→body convention as update_root_state).
        """
        self.q = q
        if head_pos is None or head_quat is None:
            self.head_pos = pickle.loads(self.redis_client.get("head_pos"))  # type: ignore
            self.head_quat = pickle.loads(self.redis_client.get("head_quat"))  # type: ignore
        else:
            self.head_pos = head_pos
            self.head_quat = head_quat

        root_pos_slam, root_quat_slam = get_root_pose_from_link(
            self.robot, self.mocap_link_id, self.q, self.head_pos, self.head_quat
        )
        root_pos_slam = np.asarray(root_pos_slam, dtype=np.float64)

        self.root_quat = np.asarray(
            self.quaternion_filter.update(imu_quat, root_quat_slam), dtype=np.float64
        )
        root_pos = np.asarray(
            self.quaternion_filter.slam_pos_to_world(root_pos_slam), dtype=np.float64
        )
        self.root_pos = root_pos.astype(np.float32)

        if not self.use_acc:
            root_vel_world, _z = self.foot_odo.estimate_velocity(
                q, dq, self.root_quat, omega
            )
        else:
            if ddq is None or root_a is None or tau is None:
                raise ValueError("ddq, root_a, and tau required when use_acc=True")
            root_vel_world, _z = self.foot_odo.estimate_velocity(
                q, dq, self.root_quat, omega, root_a, ddq, tau
            )

        # v_world from foot odom → body frame with fused root_orn (same R as update_root_state)
        root_vel_body = Rotation.from_quat(self.root_quat).inv().apply(root_vel_world)
        root_vel_body = np.asarray(root_vel_body, dtype=np.float32)
        self.root_quat = self.root_quat.astype(np.float32)

        pb.resetBasePositionAndOrientation(self.robot, root_pos, self.root_quat)
        self.root_pose = np.concatenate((root_pos, self.root_quat))
        return root_pos, self.root_quat, root_vel_body

    def get_track_site(self):
        self.set_robot_state(self.q, self.root_pose)
        target = np.zeros((3, 7))
        left_hand = pb.getLinkState(self.robot, self.eef_id[0])
        right_hand = pb.getLinkState(self.robot, self.eef_id[1])
        head = pb.getLinkState(self.robot, self.head_id)
        left_hand_pos, left_hand_quat = np.array(left_hand[0]), np.array(left_hand[1])
        right_hand_pos, right_hand_quat = np.array(right_hand[0]), np.array(right_hand[1])
        head_pos, head_quat = np.array(head[0]), np.array(head[1])
        target[0, :3] = left_hand_pos
        target[0, 3:] = left_hand_quat
        target[1, :3] = right_hand_pos
        target[1, 3:] = right_hand_quat
        target[2, :3] = head_pos
        target[2, 3:] = head_quat
        return target

    def visualize_track_sites(self, targets, only_pos=True):
        for i, frame in enumerate(self.track_sites):
            pb.resetBasePositionAndOrientation(
                frame, targets[i, :3], targets[i, 3:] if not only_pos else [0, 0, 0, 1]
            )
