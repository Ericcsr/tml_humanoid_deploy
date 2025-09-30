import numpy as np
import onnxruntime
import torch
from typing import Dict, Any, List

from tml_humanoid_deploy.utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from tml_humanoid_deploy.utils.math_utils import yaw_quat
from scipy.spatial.transform import Rotation

# base agent for deploy beyond mimic model
class BaseAgent:
    def __init__(self, onnx_model_path: str, obs_names: List[str]):
        self.onnx_model_path = onnx_model_path
        self.obs_names = obs_names
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.meta_data = self.session.get_modelmeta().custom_metadata_map
        self.ticker = 0
    
    def get_q_init(self):
        return np.zeros(29)  # default to zero position
    
    def prepare_control_signals(self, robot_state):
        raise NotImplementedError

    def prepare_obs(self, robot_state, control_signals):
        obs = []
        robot_state_keys = list(robot_state.__dict__.keys())
        for key in self.obs_names:
            if key in robot_state_keys:
                if key in ["q", "dq"]:
                    obs.append(robot_state.__dict__[key][MUJOCO_TO_ISAAC])
                else:
                    obs.append(robot_state.__dict__[key])
            else:
                obs.append(control_signals.__dict__[key])
        return np.concatenate(obs).reshape(1,-1)
    
    def get_action(self, obs):
        assert obs.shape == self.input_shape
        ort_inputs = {self.session.get_inputs()[0].name: obs}
        ort_outs = self.session.run(None, ort_inputs)
        return ort_outs[0].flatten()
