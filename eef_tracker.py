"""Forward kinematics for the G1 hands (end effectors) using MuJoCo.

Used by ``run_controller.py --eef_error`` to compute the robot's *achieved*
left/right hand world poses from proprioceptive joint angles + the estimated
root pose, so they can be compared against the reference motion's hand poses.

A dedicated (parent-process) MuJoCo model instance is used purely for FK; it is
independent of the simulation model running in the MujocoRobot child process.
"""

from __future__ import annotations

import numpy as np
import mujoco

# Hand bodies used as the left/right end effectors (match the reference motion's
# wrist bodies: body_names indices 28 / 29 = left/right_wrist_yaw_link).
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


class HandForwardKinematics:
    """Load a G1 MJCF once and forward-kinematics the two hand bodies per call."""

    def __init__(
        self,
        xml_path: str,
        left_body: str = LEFT_WRIST_BODY,
        right_body: str = RIGHT_WRIST_BODY,
    ):
        with open(xml_path, "r") as f:
            xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.left_id = self.model.body(left_body).id
        self.right_id = self.model.body(right_body).id

    def hand_world_poses(self, q, root_pos, root_orn_xyzw):
        """Return the left/right hand world poses for the given robot state.

        Args:
            q:             (29,) joint angles in MuJoCo/URDF order.
            root_pos:      (3,) root world position.
            root_orn_xyzw: (4,) root world quaternion (xyzw).

        Returns:
            (l_pos, l_quat_xyzw, r_pos, r_quat_xyzw) -- world position (3,) and
            quaternion (xyzw, 4,) for each hand.
        """
        root_orn_xyzw = np.asarray(root_orn_xyzw, dtype=np.float64)
        self.data.qpos[:3] = np.asarray(root_pos, dtype=np.float64)
        # MuJoCo root quaternion is wxyz; incoming is xyzw.
        self.data.qpos[3:7] = np.array(
            [root_orn_xyzw[3], root_orn_xyzw[0], root_orn_xyzw[1], root_orn_xyzw[2]]
        )
        self.data.qpos[7:36] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        def _pose(body_id):
            p = self.data.xpos[body_id].copy()
            qw = self.data.xquat[body_id]  # wxyz
            return p, np.array([qw[1], qw[2], qw[3], qw[0]])  # -> xyzw

        l_pos, l_quat = _pose(self.left_id)
        r_pos, r_quat = _pose(self.right_id)
        return l_pos, l_quat, r_pos, r_quat
