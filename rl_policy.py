import numpy as np
import onnxruntime
import torch
import time
import pickle
from utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from utils.math_utils import yaw_quat
from utils.storage_utils import ObsQueue, HistoryBuffer
from scipy.spatial.transform import Rotation
from pynput import keyboard
from threading import Lock

### Helper classes
class KeyboardController:
    def __init__(self):
        self.lock = Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.compliance = np.array([0.5, 0.5, 0.0])
        self.listener.start()

    def on_release(self, key):
        with self.lock:
            try:
                if key.char == "g":
                    self.compliance = np.array([0.0, 0.0, 0.0])
                elif key.char == "h":
                    self.compliance = np.array([0.5, 0.5, 0.0])
                elif key.char == "j":
                    self.compliance = np.array([0.0, 0.5, 0.0])
            except:
                pass
        print("Compliance set to:", self.compliance)
    
    def on_press(self, key):
        pass

    def get_compliance(self):
        with self.lock:
            return self.compliance.copy()

# base policy for deploy beyond mimic model
class RLBasePolicy:
    def __init__(self, onnx_model_path, obs_names):
        self.onnx_model_path = onnx_model_path
        self.obs_names = obs_names
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
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
    def __init__(self, onnx_model_path, obs_names, ref_motion_path, lookahead_steps=1, lookahead_frame_skips=1, init_at_first_frame=False):
        super().__init__(onnx_model_path, obs_names)
        

        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]
        
        self.ref_motion = np.load(ref_motion_path)
        if init_at_first_frame:
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            self.init_root_pos = self.ref_motion["body_pos_w"][0,0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]]).inv()
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

    def _control_signals_from_motion(self):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_q_pos[mid]
        ref_joint_vel = self.ref_q_vel[mid]
        ref_anchor_pos = self.ref_anchor_poses[mid]
        ref_anchor_orn = self.ref_anchor_orns[mid]
        return ref_joint_pos, ref_joint_vel, ref_anchor_pos, ref_anchor_orn

    def prepare_control_signals(self, robot_state):
        ref_joint_pos, ref_joint_vel, ref_anchor_pos, ref_anchor_orn = self._control_signals_from_motion()
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
            rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn)).as_quat()
        else:
            cmd = [ref_joint_pos, ref_joint_vel]
        
        control_signals["command"] = np.hstack(cmd).flatten()
        #print(control_signals["command"])
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["motion_anchor_pos_b"] = anchor_rot_inv.apply(rel_anchor_pos - robot_state.root_pos)
        control_signals["motion_anchor_ori_b"] = (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:,:2].flatten()
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0,0,-1]))
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
        ort_inputs = {"obs": obs.astype(np.float32)}
                      #"time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > 0:
            self.ticker = 0
        return ort_outs[0].flatten()
    
class RL3ptPolicy(RLBasePolicy):
    def __init__(self, onnx_model_path, obs_names, ref_motion_path, lookahead_steps=1, lookahead_frame_skips=1, init_at_first_frame=False):
        super().__init__(onnx_model_path, obs_names)
        self.ref_motion = np.load(ref_motion_path)

        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]
        
        if init_at_first_frame:
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            self.init_root_pos = self.ref_motion["body_pos_w"][0,0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]]).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:,0].copy()
        self.ref_vr_3point_poses = self.ref_motion["body_pos_w"][:,self.vr_3point_indices].copy() # [M, 3, 3]
        self.ref_anchor_orns = self.ref_motion["body_quat_w"][:,0][:,[1,2,3,0]].copy()
        self.ref_vr_3point_orns = self.ref_motion["body_quat_w"][:,self.vr_3point_indices][:,:,[1,2,3,0]].copy() # [M, 3, 4]
        self.ref_vr_3point_poses[:,0] += Rotation.from_quat(self.ref_vr_3point_orns[:,0], scalar_first=False).apply(self.ref_vr_3point_offsets[None,0,:])
        self.ref_vr_3point_poses[:,1] += Rotation.from_quat(self.ref_vr_3point_orns[:,1], scalar_first=False).apply(self.ref_vr_3point_offsets[None,1,:])
        self.ref_vr_3point_poses[:,2] += Rotation.from_quat(self.ref_vr_3point_orns[:,2], scalar_first=False).apply(self.ref_vr_3point_offsets[None,2,:])

        print("Motion length:", self.motion_length)
        self.anchor_id = 0
        #self.frame_vis = PoseVisualizer(axis_length=0.1, xlim=(-3,3), ylim=(-3,3), zlim=(-0.0,2))
        if lookahead_steps != 1:
            self.obs_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
            self.delay_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
    

    def get_q_init(self):
        return self.ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO]

    def _control_signals_from_motion(self):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_q_pos[mid]
        ref_joint_vel = self.ref_q_vel[mid]
        _ref_anchor_pos = self.ref_anchor_poses[mid]
        _ref_anchor_orn = self.ref_anchor_orns[mid]
        _ref_vr_3point_poses = self.ref_vr_3point_poses[mid]
        _ref_vr_3point_orns = self.ref_vr_3point_orns[mid]
        return ref_joint_pos, ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns

    def prepare_control_signals(self, robot_state):
        _ref_joint_pos, _ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns = self._control_signals_from_motion()
        # compute relative to initial frame
        rel_anchor_pos = self.init_root_heading_inv.apply(_ref_anchor_pos - self.init_root_pos)
        rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(_ref_anchor_orn)).as_quat()
        control_signals = {}
        if hasattr(self, "obs_queue"):
            this_cmd = np.stack([_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]])
            self.obs_queue.push(this_cmd)
            cmd = self.obs_queue.get_traj()
            self.delay_queue.push(np.hstack([_ref_anchor_pos, _ref_anchor_orn]))
            _ref_anchor_pos, _ref_anchor_orn = self.delay_queue[1][:3], self.delay_queue[1][3:7]
        else:
            lower_cmd = [_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]] # in isaac order
        vr_3point_pos_l = Rotation.from_quat(_ref_anchor_orn).inv().apply(_ref_vr_3point_poses - _ref_anchor_pos[None,:])
        vr_3point_orn_l = Rotation.from_quat(_ref_anchor_orn).inv() * Rotation.from_quat(_ref_vr_3point_orns)
        control_signals["lower_command"] = np.hstack(lower_cmd).flatten()
        control_signals["vr_3point_pos"] = vr_3point_pos_l.flatten()
        control_signals["vr_3point_ori"] = vr_3point_orn_l.as_quat(scalar_first=True).flatten()
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["motion_anchor_pos_b"] = anchor_rot_inv.apply(rel_anchor_pos - robot_state.root_pos)
        control_signals["motion_anchor_ori_b"] = (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:,:2].flatten()
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0,0,-1]))
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
        ort_inputs = {"obs": obs.astype(np.float32)}
                      #"time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > 0:
            self.ticker = 0
        return ort_outs[0].flatten()
    

class RLCHIPPolicy(RLBasePolicy):
    def __init__(self, onnx_model_path, obs_names, ref_motion_path, lookahead_steps=1, lookahead_frame_skips=1, hist_names=[], hist_length=10, init_at_first_frame=False):
        super().__init__(onnx_model_path, obs_names)
        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]
        
        self.ref_motion = np.load(ref_motion_path)
        self.init_at_first_frame = init_at_first_frame
        if init_at_first_frame:
            # No recentering: use world frame (robot already at first frame on terrain)
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            self.init_root_pos = self.ref_motion["body_pos_w"][0,0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]]).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:,0].copy()
        self.ref_vr_3point_poses = self.ref_motion["body_pos_w"][:,self.vr_3point_indices].copy() # [M, 3, 3]
        self.ref_anchor_orns = self.ref_motion["body_quat_w"][:,0][:,[1,2,3,0]].copy()
        self.ref_vr_3point_orns = self.ref_motion["body_quat_w"][:,self.vr_3point_indices][:,:,[1,2,3,0]].copy() # [M, 3, 4]
        self.ref_vr_3point_poses[:,0] += Rotation.from_quat(self.ref_vr_3point_orns[:,0], scalar_first=False).apply(self.ref_vr_3point_offsets[None,0,:])
        self.ref_vr_3point_poses[:,1] += Rotation.from_quat(self.ref_vr_3point_orns[:,1], scalar_first=False).apply(self.ref_vr_3point_offsets[None,1,:])
        self.ref_vr_3point_poses[:,2] += Rotation.from_quat(self.ref_vr_3point_orns[:,2], scalar_first=False).apply(self.ref_vr_3point_offsets[None,2,:])
        self.keyboard_controller = KeyboardController()
        print("Motion length:", self.motion_length)
        self.anchor_id = 0
        #self.frame_vis = PoseVisualizer(axis_length=0.1, xlim=(-3,3), ylim=(-3,3), zlim=(-0.0,2))
        if lookahead_steps != 1:
            self.obs_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
            self.delay_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
        if hist_length > 1:
            self.history_buffer = HistoryBuffer(hist_length, obs_names=hist_names, flatten=True)
    

    def get_q_init(self):
        return self.ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO]


    def _control_signals_from_motion(self):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_q_pos[mid]
        ref_joint_vel = self.ref_q_vel[mid]
        _ref_anchor_pos = self.ref_anchor_poses[mid]
        _ref_anchor_orn = self.ref_anchor_orns[mid]
        _ref_vr_3point_poses = self.ref_vr_3point_poses[mid]
        _ref_vr_3point_orns = self.ref_vr_3point_orns[mid]
        return ref_joint_pos, ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns

    def prepare_control_signals(self, robot_state):
        _ref_joint_pos, _ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns = self._control_signals_from_motion()
        # compute relative to initial frame
        rel_anchor_pos = self.init_root_heading_inv.apply(_ref_anchor_pos - self.init_root_pos)
        rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(_ref_anchor_orn)).as_quat()
        control_signals = {}
        if hasattr(self, "obs_queue"):
            this_cmd = np.stack([_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]])
            self.obs_queue.push(this_cmd)
            lower_cmd = self.obs_queue.get_traj()
            self.delay_queue.push(np.hstack([_ref_anchor_pos, _ref_anchor_orn]))
            _ref_anchor_pos, _ref_anchor_orn = self.delay_queue[1][:3], self.delay_queue[1][3:7]
        else:
            lower_cmd = [_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]] # in isaac order
        vr_3point_pos_l = Rotation.from_quat(_ref_anchor_orn).inv().apply(_ref_vr_3point_poses - _ref_anchor_pos[None,:])
        vr_3point_orn_l = Rotation.from_quat(_ref_anchor_orn).inv() * Rotation.from_quat(_ref_vr_3point_orns)
        # vr_3point_pos_l = Rotation.from_quat(self.ref_anchor_orns[0]).inv().apply(self.ref_vr_3point_poses[0] - self.ref_anchor_poses[0][None,:])
        # vr_3point_orn_l = Rotation.from_quat(self.ref_anchor_orns[0]).inv() * Rotation.from_quat(self.ref_vr_3point_orns[0])
        #control_signals["command"] = np.hstack(cmd).flatten()
        control_signals["lower_command"] = np.hstack(lower_cmd).flatten()
        control_signals["vr_3point_pos"] = vr_3point_pos_l.flatten()
        control_signals["vr_3point_ori"] = vr_3point_orn_l.as_quat(scalar_first=True).flatten()
        #print(control_signals["command"])
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["motion_anchor_pos_b"] = anchor_rot_inv.apply(rel_anchor_pos - robot_state.root_pos)
        control_signals["motion_anchor_ori_b"] = (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:,:2].flatten()
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0,0,-1]))
        control_signals["compliance"] = self.keyboard_controller.get_compliance()
        return control_signals

    def prepare_obs(self, robot_state, control_signals):
        obs = []
        robot_state_keys = list(robot_state.__dict__.keys())
        for key in self.obs_names:
            if key in robot_state_keys:
                if key in ["q", "dq"]:
                    item = robot_state.__dict__[key][MUJOCO_TO_ISAAC] - self.default_value[key]
                else:
                    item = robot_state.__dict__[key]
            else:
                item = control_signals[key]
            if hasattr(self, "history_buffer") and key in self.history_buffer.buffer_dict:
                self.history_buffer.add(key, item)
                item = self.history_buffer.get_history(key)
            obs.append(item)
        return np.concatenate(obs).reshape(1,-1)
    
    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape, f"Obs shape: {obs.shape}, input shape: {self.input_shape}"
        ort_inputs = {"obs": obs.astype(np.float32)}
                      #"time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > 0:
            self.ticker = 0
        return ort_outs[0].flatten()

class RLContactPolicy(RLBasePolicy):
    def __init__(self, onnx_model_path, obs_names, ref_motion_path, contact_labels_path, lookahead_steps=1, lookahead_frame_skips=1, hist_names=[], hist_length=10, init_at_first_frame=False):
        super().__init__(onnx_model_path, obs_names)
        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]
        
        self.ref_motion = np.load(ref_motion_path)
        self.contact_labels = np.load(contact_labels_path, allow_pickle=True).item()
        self.contact_mask = self.contact_labels["contact_mask"].astype(np.float32)
        self.init_at_first_frame = init_at_first_frame
        if init_at_first_frame:
            # No recentering: use world frame (robot already at first frame on terrain)
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            self.init_root_pos = self.ref_motion["body_pos_w"][0,0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][0,0])[[1,2,3,0]]).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:,0].copy()
        self.ref_vr_3point_poses = self.ref_motion["body_pos_w"][:,self.vr_3point_indices].copy() # [M, 3, 3]
        self.ref_anchor_orns = self.ref_motion["body_quat_w"][:,0][:,[1,2,3,0]].copy()
        self.ref_vr_3point_orns = self.ref_motion["body_quat_w"][:,self.vr_3point_indices][:,:,[1,2,3,0]].copy() # [M, 3, 4]
        self.ref_vr_3point_poses[:,0] += Rotation.from_quat(self.ref_vr_3point_orns[:,0], scalar_first=False).apply(self.ref_vr_3point_offsets[None,0,:])
        self.ref_vr_3point_poses[:,1] += Rotation.from_quat(self.ref_vr_3point_orns[:,1], scalar_first=False).apply(self.ref_vr_3point_offsets[None,1,:])
        self.ref_vr_3point_poses[:,2] += Rotation.from_quat(self.ref_vr_3point_orns[:,2], scalar_first=False).apply(self.ref_vr_3point_offsets[None,2,:])
        self.keyboard_controller = KeyboardController()
        print("Motion length:", self.motion_length)
        self.anchor_id = 0
        #self.frame_vis = PoseVisualizer(axis_length=0.1, xlim=(-3,3), ylim=(-3,3), zlim=(-0.0,2))
        if lookahead_steps != 1:
            self.obs_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
            self.delay_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
        if hist_length > 1:
            self.history_buffer = HistoryBuffer(hist_length, obs_names=hist_names, flatten=True)
    

    def get_q_init(self):
        return self.ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO]


    def _control_signals_from_motion(self):
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        ref_joint_pos = self.ref_q_pos[mid]
        ref_joint_vel = self.ref_q_vel[mid]
        _ref_anchor_pos = self.ref_anchor_poses[mid]
        _ref_anchor_orn = self.ref_anchor_orns[mid]
        _ref_vr_3point_poses = self.ref_vr_3point_poses[mid]
        _ref_vr_3point_orns = self.ref_vr_3point_orns[mid]
        return ref_joint_pos, ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns

    def prepare_control_signals(self, robot_state):
        _ref_joint_pos, _ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns = self._control_signals_from_motion()
        # compute relative to initial frame
        rel_anchor_pos = self.init_root_heading_inv.apply(_ref_anchor_pos - self.init_root_pos)
        rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(_ref_anchor_orn)).as_quat()
        control_signals = {}
        if hasattr(self, "obs_queue"):
            this_cmd = np.stack([_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]])
            self.obs_queue.push(this_cmd)
            lower_cmd = self.obs_queue.get_traj()
            self.delay_queue.push(np.hstack([_ref_anchor_pos, _ref_anchor_orn]))
            _ref_anchor_pos, _ref_anchor_orn = self.delay_queue[1][:3], self.delay_queue[1][3:7]
        else:
            lower_cmd = [_ref_joint_pos[self.lower_joint_indices], _ref_joint_vel[self.lower_joint_indices]] # in isaac order
        vr_3point_pos_l = Rotation.from_quat(_ref_anchor_orn).inv().apply(_ref_vr_3point_poses - _ref_anchor_pos[None,:])
        vr_3point_orn_l = Rotation.from_quat(_ref_anchor_orn).inv() * Rotation.from_quat(_ref_vr_3point_orns)
        # vr_3point_pos_l = Rotation.from_quat(self.ref_anchor_orns[0]).inv().apply(self.ref_vr_3point_poses[0] - self.ref_anchor_poses[0][None,:])
        # vr_3point_orn_l = Rotation.from_quat(self.ref_anchor_orns[0]).inv() * Rotation.from_quat(self.ref_vr_3point_orns[0])
        #control_signals["command"] = np.hstack(cmd).flatten()
        control_signals["lower_command"] = np.hstack(lower_cmd).flatten()
        control_signals["vr_3point_pos"] = vr_3point_pos_l.flatten()
        control_signals["vr_3point_ori"] = vr_3point_orn_l.as_quat(scalar_first=True).flatten()
        #print(control_signals["command"])
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["motion_anchor_pos_b"] = anchor_rot_inv.apply(rel_anchor_pos - robot_state.root_pos)
        control_signals["motion_anchor_ori_b"] = (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:,:2].flatten()
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0,0,-1]))
        control_signals["compliance"] = self.keyboard_controller.get_compliance()
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length-1
        control_signals["contact_mask"] = self.contact_mask[mid]
        return control_signals

    def prepare_obs(self, robot_state, control_signals):
        obs = []
        robot_state_keys = list(robot_state.__dict__.keys())
        for key in self.obs_names:
            if key in robot_state_keys:
                if key in ["q", "dq"]:
                    item = robot_state.__dict__[key][MUJOCO_TO_ISAAC] - self.default_value[key]
                else:
                    item = robot_state.__dict__[key]
            else:
                item = control_signals[key]
            if hasattr(self, "history_buffer") and key in self.history_buffer.buffer_dict:
                self.history_buffer.add(key, item)
                item = self.history_buffer.get_history(key)
            obs.append(item)
        return np.concatenate(obs).reshape(1,-1)
    
    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape, f"Obs shape: {obs.shape}, input shape: {self.input_shape}"
        ort_inputs = {"obs": obs.astype(np.float32)}
                      #"time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > 0:
            self.ticker = 0
        return ort_outs[0].flatten()