import numpy as np
import onnxruntime
import torch
from typing import List, Dict, Any

from tml_humanoid_deploy.utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from tml_humanoid_deploy.utils.math_utils import yaw_quat
from scipy.spatial.transform import Rotation
from tml_humanoid_deploy.agents.base_agent import BaseAgent


class RLBMAgent(BaseAgent):
    def __init__(self, onnx_model_path, obs_names, ref_motion_path):
        super().__init__(onnx_model_path, obs_names)
        self.ref_motion = np.load(ref_motion_path)
        self.init_root_pos = self.ref_motion["body_pos_w"][0,0]
        self.init_root_pos[2] = 0  # set initial height to 0
        self.init_root_heading = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]])

        self.init_robot_state = None

        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        print("Motion length:", self.motion_length)
        self.anchor_id = 0
    
    def get_q_init(self):
        return self.ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO]
    
    def prepare_control_signals(self, robot_state):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_motion["joint_pos"][mid] # don't matter that much
        ref_joint_vel = self.ref_motion["joint_vel"][mid]
        ref_anchor_pos = self.ref_motion["body_pos_w"][mid, 0]
        ref_anchor_orn = self.ref_motion["body_quat_w"][mid, 0][[1,2,3,0]]
        # compute relative to initial frame
        if self.init_robot_state is None:
            self.init_robot_state = (robot_state.root_pos.copy(), Rotation.from_quat(robot_state.root_orn.copy()))

        rel_anchor_pos = self.init_root_heading.inv().apply(ref_anchor_pos - self.init_root_pos)
        rel_anchor_orn = (self.init_root_heading.inv() * Rotation.from_quat(ref_anchor_orn)).as_quat()
        #rel_anchor_pos = self.init_robot_state[1].apply(self.init_robot_state[0] + rel_anchor_pos)
        #rel_anchor_orn = (self.init_robot_state[1] * Rotation.from_quat(rel_anchor_orn)).as_quat()

        control_signals = {}
        control_signals["command"] = np.hstack([ref_joint_pos, ref_joint_vel])
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["motion_anchor_pos_b"] = anchor_rot_inv.apply(rel_anchor_pos - robot_state.root_pos)
        control_signals["motion_anchor_ori_b"] = (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:,:2].flatten()
        return control_signals

    def prepare_obs(self, robot_state, control_signals):
        obs = []
        robot_state_keys = list(robot_state.__dict__.keys())
        for key in self.obs_names:
            if key in robot_state_keys:
                if key in ["q", "dq"]:
                    obs.append(robot_state.__dict__[key][MUJOCO_TO_ISAAC] - self.default_value[key])
                else:
                    obs.append(robot_state.__dict__[key])
            else:
                obs.append(control_signals[key])
        return np.concatenate(obs).reshape(1,-1)
    
    def get_action(self, obs):
        assert obs.shape == self.input_shape
        ort_inputs = {"obs": obs.astype(np.float32), 
                      "time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        self.ticker += 1
        return ort_outs[0].flatten()
