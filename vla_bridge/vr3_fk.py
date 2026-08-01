"""
MuJoCo forward-kinematics of the 3 "VR" bodies (left/right wrist-yaw + torso) that
the tracking controller consumes as ``vr_3point_*``.

Ported from ``stair-rendering/stair_sim_eval.py`` (``fk_vr3`` + VR3_BODY_NAMES /
VR3_OFFSETS). The controller (``RLContactPolicy``/``replay_server``) originally
reads these from the reference clip's ``body_pos_w[:, [28,29,9]]`` with the same
offsets; a GR00T chunk carries only joints + root, so the bridge re-derives the
3-point world poses by FK on the deploy MuJoCo scene (30 robot bodies) — matching
the sim_eval path exactly.
"""

import os

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

NUM_JOINTS = 29

# Wrist-yaw + torso, with the same per-body offsets rl_policy/replay_server add.
VR3_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
VR3_OFFSETS = np.array(
    [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]], np.float64
)


class VR3ForwardKinematics:
    """Loads a deploy MuJoCo scene once and FK's the 3-point poses per call.

    The scene XML ``<include>``s ``assets/g1/...`` relative to the deploy root, so
    the model is compiled from the file's TEXT with cwd at ``deploy_root`` (same
    trick convert_dataset_to_ref_motion / sim_eval use).
    """

    def __init__(self, deploy_root, mujoco_xml_path):
        deploy_root = os.path.abspath(os.path.expanduser(deploy_root))
        xml = mujoco_xml_path if os.path.isabs(mujoco_xml_path) else os.path.join(
            deploy_root, mujoco_xml_path
        )
        old = os.getcwd()
        try:
            os.chdir(deploy_root)
            self.model = mujoco.MjModel.from_xml_string(open(xml).read())
        finally:
            os.chdir(old)
        if self.model.nq != 3 + 4 + NUM_JOINTS:
            raise ValueError(f"model.nq={self.model.nq}, expected {3 + 4 + NUM_JOINTS}")
        self.data = mujoco.MjData(self.model)
        self.body_ids = {n: self.model.body(n).id for n in VR3_BODY_NAMES}

    def __call__(self, root_pos, root_quat_wxyz, joints_mujoco):
        """FK one frame. Returns (poses (3,3), orns_xyzw (3,4)) for the 3 points."""
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat_wxyz          # MuJoCo free joint is wxyz
        self.data.qpos[7:7 + NUM_JOINTS] = joints_mujoco
        mujoco.mj_forward(self.model, self.data)
        poses = np.stack([self.data.xpos[self.body_ids[n]].copy() for n in VR3_BODY_NAMES])
        orns = np.stack(
            [self.data.xquat[self.body_ids[n]][[1, 2, 3, 0]].copy() for n in VR3_BODY_NAMES]
        )  # xquat is wxyz -> xyzw
        for i in range(3):
            poses[i] += R.from_quat(orns[i]).apply(VR3_OFFSETS[i])
        return poses.astype(np.float32), orns.astype(np.float32)
