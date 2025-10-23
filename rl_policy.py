import numpy as np
import onnxruntime
import torch
import time
from utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from utils.math_utils import yaw_quat
from utils.storage_utils import ObsQueue
from scipy.spatial.transform import Rotation

# base policy for deploy beyond mimic model
class RLBasePolicy:
    def __init__(self, onnx_model_path, obs_names):
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


class RLBMPolicy(RLBasePolicy):
    def __init__(self, onnx_model_path, obs_names, ref_motion_path, lookahead_steps=1, lookahead_frame_skips=1):
        super().__init__(onnx_model_path, obs_names)
        self.ref_motion = np.load(ref_motion_path)
        self.init_root_pos = self.ref_motion["body_pos_w"][0,0]
        self.init_root_pos[2] = 0  # set initial height to 0
        self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]]).inv()

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
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:,0].copy()
        self.ref_anchor_orns = self.ref_motion["body_quat_w"][:,0][:,[1,2,3,0]].copy()
        print("Motion length:", self.motion_length)
        self.anchor_id = 0
        if lookahead_steps != 1:
            self.obs_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
            self.delay_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
    

    def get_q_init(self):
        return self.ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO]

    def prepare_control_signals(self, robot_state):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_q_pos[mid]
        ref_joint_vel = self.ref_q_vel[mid]
        ref_anchor_pos = self.ref_anchor_poses[mid]
        ref_anchor_orn = self.ref_anchor_orns[mid]
        # compute relative to initial frame
        rel_anchor_pos = self.init_root_heading_inv.apply(ref_anchor_pos - self.init_root_pos)
        rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn)).as_quat()
        control_signals = {}
        if hasattr(self, "obs_queue"):
            this_cmd = np.stack([ref_joint_pos, ref_joint_vel])
            self.obs_queue.push(this_cmd)
            cmd = self.obs_queue.get_traj()
            self.delay_queue.push(np.hstack([ref_anchor_pos, ref_anchor_orn]))
            ref_anchor_pos, ref_anchor_orn = self.delay_queue[1][:3], self.delay_queue[1][3:7]
        else:
            cmd = [ref_joint_pos, ref_joint_vel]
        
        control_signals["command"] = np.hstack(cmd).flatten()
        #print(control_signals["command"])
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
    
    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape
        ort_inputs = {"obs": obs.astype(np.float32), 
                      "time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > 0:
            self.ticker = 0
        return ort_outs[0].flatten()