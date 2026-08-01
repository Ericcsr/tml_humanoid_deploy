import numpy as np
import onnxruntime
import torch
import time
import pickle
import redis
import pybullet as pb
import os
from utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from utils.math_utils import yaw_quat, yaw_quat_xyzw
from utils.storage_utils import ObsQueue, HistoryBuffer
from utils.redis_utils import REDIS_IP, REDIS_PORT
from scipy.spatial.transform import Rotation
from pynput import keyboard
from threading import Lock

import sys
import numpy.core.multiarray as multiarray

# Redirect the specific path the pickle is looking for
sys.modules['numpy._core.multiarray'] = multiarray


def clamp_ref_motion_start_index(start: int, motion_length: int) -> int:
    """Clamp start frame index to [0, motion_length - 1]."""
    if motion_length <= 0:
        return 0
    s = int(start)
    if s < 0:
        return 0
    if s >= motion_length:
        return motion_length - 1
    return s


def _parse_slow_motion_params(
    slow_motion_end_frame,
    slow_down_times,
    motion_length: int,
):
    """Return validated (end_frame, repeat_times), or (None, None) when disabled."""
    if motion_length <= 0:
        return None, None
    if slow_motion_end_frame is None or slow_down_times is None:
        return None, None
    try:
        end_frame = int(slow_motion_end_frame)
        repeat_times = int(slow_down_times)
    except (TypeError, ValueError):
        return None, None
    if end_frame <= 0 or repeat_times <= 1:
        return None, None
    end_frame = min(end_frame, motion_length)
    if end_frame <= 0:
        return None, None
    return end_frame, repeat_times


def _stretch_prefix_along_time(arr: np.ndarray, end_frame: int, repeat_times: int) -> np.ndarray:
    """Repeat frames [0:end_frame) along axis-0 by repeat_times."""
    prefix = np.repeat(arr[:end_frame], repeat_times, axis=0)
    return np.concatenate((prefix, arr[end_frame:]), axis=0)


def load_ref_motion_with_optional_slowdown(
    ref_motion_path: str,
    slow_motion_end_frame=None,
    slow_down_times=None,
):
    """
    Load ref motion (.npz), optionally stretching first end_frame frames by repeat_times.
    Returns either the original np.load object or a dict[str, np.ndarray] with stretched data.
    """
    ref_motion = np.load(ref_motion_path)
    motion_length = int(ref_motion["joint_pos"].shape[0])
    end_frame, repeat_times = _parse_slow_motion_params(
        slow_motion_end_frame, slow_down_times, motion_length
    )
    if end_frame is None:
        return ref_motion

    stretched = {}
    for key in ref_motion.files:
        value = np.asarray(ref_motion[key])
        if value.ndim > 0 and value.shape[0] == motion_length:
            stretched[key] = _stretch_prefix_along_time(value, end_frame, repeat_times)
        else:
            stretched[key] = value.copy()
    print(
        f"[ref_motion] slow prefix enabled: first {end_frame} frames x{repeat_times} "
        f"(length {motion_length} -> {stretched['joint_pos'].shape[0]})",
        flush=True,
    )
    return stretched


def stretch_contact_mask_prefix_if_enabled(
    contact_mask: np.ndarray,
    slow_motion_end_frame=None,
    slow_down_times=None,
) -> np.ndarray:
    """Apply the same slow-prefix stretching on a (T, C) contact mask."""
    mask = np.asarray(contact_mask, dtype=np.float32)
    if mask.ndim != 2:
        return mask
    end_frame, repeat_times = _parse_slow_motion_params(
        slow_motion_end_frame, slow_down_times, int(mask.shape[0])
    )
    if end_frame is None:
        return mask
    stretched = _stretch_prefix_along_time(mask, end_frame, repeat_times).astype(np.float32)
    print(
        f"[contact_mask] slow prefix enabled: first {end_frame} frames x{repeat_times} "
        f"(length {mask.shape[0]} -> {stretched.shape[0]})",
        flush=True,
    )
    return stretched


def expand_4way_contact_to_8(m4: np.ndarray) -> np.ndarray:
    """
    4-way [Lfoot, Rfoot, Lwrist, Rwrist] -> 8-way
    [Lfoot_env, Lfoot_obj, Rfoot_env, Rfoot_obj, Lwrist_env, Lwrist_obj, Rwrist_env, Rwrist_obj]
    Default split: foot values -> env; wrist values -> object; the paired channel is 0.
    """
    m4 = np.asarray(m4, dtype=np.float32)
    lead = m4.ndim - 1
    if m4.shape[lead] != 4:
        raise ValueError(f"expand_4way_contact_to_8: expected 4 contact channels, got {m4.shape[lead]}")
    sl = (slice(None),) * lead
    m8 = np.zeros(m4.shape[:lead] + (8,), dtype=np.float32)
    m8[sl + (0,)] = m4[sl + (0,)]
    m8[sl + (2,)] = m4[sl + (1,)]
    m8[sl + (5,)] = m4[sl + (2,)]
    m8[sl + (7,)] = m4[sl + (3,)]
    return m8


def expand_5way_contact_to_10(m5: np.ndarray) -> np.ndarray:
    """
    5-way stored labels -> 10-way obs: 4-way [Lfoot, Rfoot, Lwrist, Rwrist] expanded to 8-way,
    then 2-way pelvis/seat [env, obj] appended (column 4 -> env; obj channel 0), same split
    pattern as expand_4way_contact_to_8 for a single logical contact.
    """
    m5 = np.asarray(m5, dtype=np.float32)
    lead = m5.ndim - 1
    if m5.shape[lead] != 5:
        raise ValueError(f"expand_5way_contact_to_10: expected 5 contact channels, got {m5.shape[lead]}")
    sl = (slice(None),) * lead
    m4 = m5[sl + (slice(0, 4),)].copy()
    seat = m5[sl + (4,)]
    m8 = expand_4way_contact_to_8(m4)
    m10 = np.zeros(m5.shape[:lead] + (10,), dtype=np.float32)
    m10[sl + (slice(0, 8),)] = m8
    m10[sl + (8,)] = seat
    m10[sl + (9,)] = np.float32(0.0)
    return m10


def pad_4way_contact_to_5(m4: np.ndarray) -> np.ndarray:
    """Append a zero column: (..., 4) -> (..., 5) for pelvis/seat channel (unused when sourcing 4-way data)."""
    m4 = np.asarray(m4, dtype=np.float32)
    lead = m4.ndim - 1
    if m4.shape[lead] != 4:
        raise ValueError(f"pad_4way_contact_to_5: expected 4 contact channels, got {m4.shape[lead]}")
    z = np.zeros(m4.shape[:lead] + (1,), dtype=np.float32)
    return np.concatenate([m4, z], axis=lead)


def expand_4way_limb_to_10way_with_zero_seat(m4: np.ndarray) -> np.ndarray:
    """(..., 4) 4-way limbs -> (..., 10): 8-way limb channels + [0, 0] pelvis/seat (2-way)."""
    m8 = expand_4way_contact_to_8(np.asarray(m4, dtype=np.float32))
    lead = m8.ndim - 1
    z2 = np.zeros(m8.shape[:lead] + (2,), dtype=np.float32)
    return np.concatenate([m8, z2], axis=lead)


def reduce_8way_contact_to_4(m8: np.ndarray) -> np.ndarray:
    """
    8-way -> 4-way by max(env, obj) per limb (Lfoot, Rfoot, Lwrist, Rwrist).
    """
    m8 = np.asarray(m8, dtype=np.float32)
    lead = m8.ndim - 1
    if m8.shape[lead] != 8:
        raise ValueError(f"reduce_8way_contact_to_4: expected 8 contact channels, got {m8.shape[lead]}")
    sl = (slice(None),) * lead
    m4 = np.empty(m8.shape[:lead] + (4,), dtype=np.float32)
    m4[sl + (0,)] = np.maximum(m8[sl + (0,)], m8[sl + (1,)])
    m4[sl + (1,)] = np.maximum(m8[sl + (2,)], m8[sl + (3,)])
    m4[sl + (2,)] = np.maximum(m8[sl + (4,)], m8[sl + (5,)])
    m4[sl + (3,)] = np.maximum(m8[sl + (6,)], m8[sl + (7,)])
    return m4


def normalize_contact_mask_labels(
    raw: np.ndarray,
    *,
    use_8way_contact: bool,
) -> np.ndarray:
    """
    Load (T,4) or (T,8) to match deploy mode: 4-way policy vs 8-way.
    4+8 mix: 4 with use_8way -> expand; 8 without use_8way -> max-pool to 4.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"contact_mask must be 2D (T, C), got shape {raw.shape}")
    c = raw.shape[1]
    if use_8way_contact:
        if c == 8:
            return raw
        if c == 4:
            return expand_4way_contact_to_8(raw)
        raise ValueError(
            f"contact_mask: use_8way_contact is True, expected 4 or 8 columns, got {c}"
        )
    if c == 4:
        return raw
    if c == 8:
        return reduce_8way_contact_to_4(raw)
    raise ValueError(
        f"contact_mask: use_8way_contact is False, expected 4 or 8 columns, got {c}"
    )


def normalize_contact_mask_for_10way(raw: np.ndarray) -> np.ndarray:
    """
    use_10way_contact: file (T,5) [4-way limbs + pelvis/seat scalar] -> (T,10), or passthrough (T,10).
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"contact_mask must be 2D (T, C), got shape {raw.shape}")
    c = raw.shape[1]
    if c == 10:
        return raw
    if c == 5:
        return expand_5way_contact_to_10(raw)
    raise ValueError(
        "contact_mask: use_10way_contact is True, expected 5 columns "
        f"([Lfoot,Rfoot,Lwrist,Rwrist,pelvis_seat]) or 10 columns, got {c}"
    )


def _compute_initial_vr_3point_local_from_pybullet(
    default_q_isaac,
    urdf_path,
    vr_3point_link_names,
    vr_3point_offsets,
    anchor_link_name,
):
    """Compute initial local 3-point targets (pos/orientation) from FK."""
    client = pb.connect(pb.DIRECT)
    try:
        robot = pb.loadURDF(urdf_path, useFixedBase=True, physicsClientId=client)
        num_joints = pb.getNumJoints(robot, physicsClientId=client)
        revolute_joints = []
        link_name_to_joint = {}
        for j in range(num_joints):
            ji = pb.getJointInfo(robot, j, physicsClientId=client)
            if ji[2] == pb.JOINT_REVOLUTE:
                revolute_joints.append(j)
            link_name_to_joint[ji[12].decode("utf-8")] = j

        q = np.asarray(default_q_isaac, dtype=np.float64).reshape(-1)
        if len(revolute_joints) != q.size:
            raise ValueError(f"FK mismatch: revolute joints {len(revolute_joints)} vs q size {q.size}")
        pb.resetJointStatesMultiDof(
            robot,
            revolute_joints,
            targetValues=q.reshape(-1, 1),
            physicsClientId=client,
        )

        anchor_jid = link_name_to_joint[anchor_link_name]
        anchor_state = pb.getLinkState(robot, anchor_jid, computeForwardKinematics=True, physicsClientId=client)
        anchor_pos = np.array(anchor_state[4], dtype=np.float64)
        anchor_orn = np.array(anchor_state[5], dtype=np.float64)  # xyzw
        anchor_rot_inv = Rotation.from_quat(anchor_orn).inv()

        pos_l = []
        orn_l = []
        for i, link_name in enumerate(vr_3point_link_names):
            jid = link_name_to_joint[link_name]
            state = pb.getLinkState(robot, jid, computeForwardKinematics=True, physicsClientId=client)
            link_pos = np.array(state[4], dtype=np.float64)
            link_orn = np.array(state[5], dtype=np.float64)  # xyzw
            world_offset = Rotation.from_quat(link_orn).apply(vr_3point_offsets[i])
            link_pos_offset = link_pos + world_offset
            pos_l.append(anchor_rot_inv.apply(link_pos_offset - anchor_pos))
            rel_orn = anchor_rot_inv * Rotation.from_quat(link_orn)
            orn_l.append(rel_orn.as_quat(scalar_first=True))

        return (
            np.asarray(pos_l, dtype=np.float32).reshape(-1),
            np.asarray(orn_l, dtype=np.float32).reshape(-1),
        )
    finally:
        pb.disconnect(client)

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
        #print("Compliance set to:", self.compliance)
    
    def on_press(self, key):
        pass

    def get_compliance(self):
        with self.lock:
            return self.compliance.copy()

# base policy for deploy beyond mimic model
class RLBasePolicy:
    def __init__(self, onnx_model_path, obs_names, use_sim=False):
        self.onnx_model_path = onnx_model_path
        self.obs_names = obs_names
        if use_sim:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
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
    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        init_at_first_frame=False,
        ref_motion_start_index=0,
        slow_motion_end_frame=None,
        slow_down_times=None,
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)
        

        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]
        
        self.ref_motion = load_ref_motion_with_optional_slowdown(
            ref_motion_path,
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_motion_start_index = clamp_ref_motion_start_index(ref_motion_start_index, self.motion_length)
        self.ticker = self.ref_motion_start_index
        if init_at_first_frame:
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            si = self.ref_motion_start_index
            self.init_root_pos = self.ref_motion["body_pos_w"][si, 0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat(self.ref_motion["body_quat_w"][si, 0])[[1,2,3,0]]).inv()
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
        si = self.ref_motion_start_index
        return self.ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO]

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
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()
    
class RL3ptPolicy(RLBasePolicy):
    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        init_at_first_frame=False,
        ref_motion_start_index=0,
        slow_motion_end_frame=None,
        slow_down_times=None,
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)
        self.ref_motion = load_ref_motion_with_optional_slowdown(
            ref_motion_path,
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )

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
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_motion_start_index = clamp_ref_motion_start_index(ref_motion_start_index, self.motion_length)
        self.ticker = self.ref_motion_start_index

        if init_at_first_frame:
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            si = self.ref_motion_start_index
            self.init_root_pos = self.ref_motion["body_pos_w"][si, 0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(
                yaw_quat(self.ref_motion["body_quat_w"][si, 0])[[1, 2, 3, 0]]
            ).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
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
        si = self.ref_motion_start_index
        return self.ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO]

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
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()
    

class RLCHIPPolicy(RLBasePolicy):
    DEFAULT_Q_POSE = [-0.312, -0.312,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.669,  0.669,  0.2  ,  0.2  , -0.363, -0.363,  0.2  ,
       -0.2  ,  0.   ,  0.   ,  0.   ,  0.   ,  0.6  ,  0.6  ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ]
    ACTION_SCALE = [0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.548, 0.548, 0.439,
       0.351, 0.351, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
       0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
       0.075, 0.075]
    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=[],
        hist_length=10,
        init_at_first_frame=False,
        ref_motion_start_index=0,
        slow_motion_end_frame=None,
        slow_down_times=None,
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)
        default_joint_pos = self.meta_data["default_joint_pos"].split(",") if "default_joint_pos" in self.meta_data else RLCHIPPolicy.DEFAULT_Q_POSE
        action_scale = self.meta_data["action_scale"].split(",") if "action_scale" in self.meta_data else RLCHIPPolicy.ACTION_SCALE
        if "default_joint_pos" in self.meta_data:
            default_joint_pos = np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")])
        if "action_scale" in self.meta_data:
            action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        self.default_value = {
            "q": np.array(default_joint_pos),
            "dq": np.zeros(29)
        }
        self.action_scale = np.array(action_scale)
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]
        
        self.ref_motion = load_ref_motion_with_optional_slowdown(
            ref_motion_path,
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
        self.init_at_first_frame = init_at_first_frame
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_motion_start_index = clamp_ref_motion_start_index(ref_motion_start_index, self.motion_length)
        self.ticker = self.ref_motion_start_index
        if init_at_first_frame:
            # No recentering: use world frame (robot already at first frame on terrain)
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            si = self.ref_motion_start_index
            self.init_root_pos = self.ref_motion["body_pos_w"][si, 0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(
                yaw_quat(self.ref_motion["body_quat_w"][si, 0])[[1, 2, 3, 0]]
            ).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
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
        si = self.ref_motion_start_index
        return self.ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO]


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
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()

class RLGlobalCHIPPolicy(RLCHIPPolicy):
    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=[],
        hist_length=10,
        init_at_first_frame=False,
        ref_motion_start_index=0,
        slow_motion_end_frame=None,
        slow_down_times=None,
    ):
        super().__init__(
            onnx_model_path,
            obs_names,
            ref_motion_path,
            use_sim,
            lookahead_steps,
            lookahead_frame_skips,
            hist_names,
            hist_length,
            init_at_first_frame,
            ref_motion_start_index=ref_motion_start_index,
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )

    def prepare_control_signals(self, robot_state):
        _ref_joint_pos, _ref_joint_vel, _ref_anchor_pos, _ref_anchor_orn, _ref_vr_3point_poses, _ref_vr_3point_orns = self._control_signals_from_motion()
        # compute relative to initial frame
        control_signals = {}
        if hasattr(self, "obs_queue"):
            this_cmd = np.hstack([_ref_vr_3point_poses, _ref_vr_3point_orns])
            self.obs_queue.push(this_cmd)
            cmd = np.stack(self.obs_queue.get_traj())
        else:
            cmd = np.hstack([_ref_vr_3point_poses, _ref_vr_3point_orns])

        # Get current robot 3point positions in global frame
        heading = yaw_quat_xyzw(robot_state.root_orn)
        vr_3point_pos_l = (Rotation.from_quat(heading).inv().apply(cmd[:,:,:3].reshape(-1,3) - robot_state.root_pos[None,:])).reshape(-1,3,3)
        head_orn_l = (Rotation.from_quat(heading).inv() * Rotation.from_quat(cmd[:,2,3:7]))
        control_signals["vr_3point_pos"] = vr_3point_pos_l.flatten()
        control_signals["head_ori"] = head_orn_l.as_quat(scalar_first=True).flatten()
        #print(control_signals["command"])
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0,0,-1]))
        control_signals["compliance"] = self.keyboard_controller.get_compliance()
        return control_signals

    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape, f"Obs shape: {obs.shape}, input shape: {self.input_shape}"
        ort_inputs = {"obs_dict": obs.astype(np.float32)}
                      #"time_step": np.array([[0.0]], dtype=np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()


class RLContactPolicy(RLBasePolicy):
    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        contact_labels_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=[],
        hist_length=10,
        init_at_first_frame=False,
        zero_foot_contact_on_load=False,
        use_8way_contact=False,
        use_10way_contact=False,
        use_5dim_contact_from_4dim=False,
        ref_motion_start_index=0,
        slow_motion_end_frame=None,
        slow_down_times=None,
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)
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
        self.use_10way_contact = bool(use_10way_contact)
        self.use_5dim_contact_from_4dim = bool(use_5dim_contact_from_4dim)
        if self.use_5dim_contact_from_4dim and (not self.use_10way_contact) and use_8way_contact:
            raise ValueError(
                "use_5dim_contact_from_4dim (5-dim output) requires use_8way_contact: false, "
                "or enable use_10way_contact for 10-dim output with zero seat."
            )
        self.use_8way_contact = bool(use_8way_contact) or self.use_10way_contact
        if self.use_10way_contact:
            self.contact_dim = 10
        elif self.use_5dim_contact_from_4dim:
            self.contact_dim = 5
        else:
            self.contact_dim = 8 if use_8way_contact else 4

        self.ref_motion = load_ref_motion_with_optional_slowdown(
            ref_motion_path,
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
        tlen = int(self.ref_motion["joint_pos"].shape[0])
        if contact_labels_path != "default":
            self.contact_labels = np.load(contact_labels_path, allow_pickle=True).item()
            self.contact_mask = self.contact_labels["contact_mask"].astype(np.float32)
            #self.contact_mask[:55,2:4] = 0.0
            self.contact_mask = stretch_contact_mask_prefix_if_enabled(
                self.contact_mask,
                slow_motion_end_frame=slow_motion_end_frame,
                slow_down_times=slow_down_times,
            )
            if self.contact_mask.ndim != 2:
                raise ValueError(
                    f"contact_mask must be 2D (T, C), got shape {self.contact_mask.shape}"
                )
            tm, tc = self.contact_mask.shape[0], self.contact_mask.shape[1]
            if self.use_10way_contact and self.use_5dim_contact_from_4dim:
                if tc not in (4, 5, 8, 10):
                    raise ValueError(
                        "contact_mask: use_10way_contact + use_5dim_contact_from_4dim expects "
                        f"4, 5, 8, or 10 columns, got {tc} (T={tm})"
                    )
            elif self.use_10way_contact:
                if tc not in (5, 10):
                    raise ValueError(
                        f"contact_mask: use_10way_contact is True, expected 5 or 10 columns, got {tc} (T={tm})"
                    )
            else:
                if tc == 5 and use_8way_contact:
                    raise ValueError(
                        "contact_mask has 5 columns: use use_10way_contact to expand to 10-way, "
                        "or set use_8way_contact false to keep a 5-dim contact_mask."
                    )
                if self.use_5dim_contact_from_4dim:
                    if tc == 5:
                        raise ValueError(
                            "use_5dim_contact_from_4dim expects 4- or 8-column 4-way data; "
                            "use a 4-column file or disable this flag for native 5-column labels."
                        )
                    if tc not in (4, 8):
                        raise ValueError(
                            f"use_5dim_contact_from_4dim: expected 4 or 8 columns, got {tc} (T={tm})"
                        )
                elif tc not in (4, 5, 8):
                    raise ValueError(
                        f"contact_mask: expected 4, 5, or 8 columns, got {tc} (T={tm})"
                    )
            if tm < tlen:
                pad = np.zeros((tlen - tm, tc), dtype=np.float32)
                self.contact_mask = np.vstack([self.contact_mask, pad])
            elif tm > tlen:
                self.contact_mask = self.contact_mask[:tlen]
            #self.contact_mask[35:,2:] = 1
            if self.use_10way_contact and self.use_5dim_contact_from_4dim:
                m = self.contact_mask
                if tc == 4:
                    self.contact_mask = expand_4way_limb_to_10way_with_zero_seat(m)
                elif tc == 5:
                    self.contact_mask = expand_4way_limb_to_10way_with_zero_seat(m[:, :4])
                elif tc == 8:
                    if use_8way_contact:
                        z2 = np.zeros((m.shape[0], 2), dtype=np.float32)
                        self.contact_mask = np.hstack([m, z2])
                    else:
                        m4 = reduce_8way_contact_to_4(m)
                        self.contact_mask = expand_4way_limb_to_10way_with_zero_seat(m4)
                else:  # tc == 10
                    z2 = np.zeros((m.shape[0], 2), dtype=np.float32)
                    self.contact_mask = np.hstack([m[:, :8].copy(), z2])
                self.contact_dim = 10
            elif self.use_10way_contact:
                self.contact_mask = normalize_contact_mask_for_10way(self.contact_mask)
            elif self.use_5dim_contact_from_4dim:
                m = self.contact_mask
                if tc == 8:
                    m = reduce_8way_contact_to_4(m)
                self.contact_mask = pad_4way_contact_to_5(m)
                self.contact_dim = 5
            elif tc == 5:
                self.contact_dim = 5
            else:
                self.contact_mask = normalize_contact_mask_labels(
                    self.contact_mask, use_8way_contact=use_8way_contact
                )
            if zero_foot_contact_on_load and self.contact_mask.ndim == 2:
                self.contact_mask = self.contact_mask.copy()
                if self.use_10way_contact:
                    self.contact_mask[:, 0:4] = 0.0
                    self.contact_mask[:, 8:10] = 0.0
                elif use_8way_contact:
                    self.contact_mask[:, 0:4] = 0.0
                elif self.contact_mask.shape[1] >= 2:
                    self.contact_mask[:, :2] = 0.0
                print(
                    "[RLContactPolicy] zero_foot_contact_on_load: selected contact channels set to 0",
                    flush=True,
                )
        else:
            self.contact_mask = np.zeros((tlen, self.contact_dim), dtype=np.float32)
        print(
            f"[RLContactPolicy] contact_mask (T, C)={self.contact_mask.shape} "
            f"use_8way_contact={use_8way_contact} use_10way_contact={self.use_10way_contact} "
            f"use_5dim_contact_from_4dim={self.use_5dim_contact_from_4dim}",
            flush=True,
        )
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_motion_start_index = clamp_ref_motion_start_index(ref_motion_start_index, self.motion_length)
        self.ticker = self.ref_motion_start_index
        self.init_at_first_frame = init_at_first_frame
        if init_at_first_frame:
            # No recentering: use world frame (robot already at first frame on terrain)
            self.init_root_pos = np.zeros(3)
            self.init_root_heading_inv = Rotation.identity()
        else:
            si = self.ref_motion_start_index
            self.init_root_pos = self.ref_motion["body_pos_w"][si, 0].copy()
            self.init_root_pos[2] = 0  # set initial height to 0 (flat ground)
            self.init_root_heading_inv = Rotation.from_quat(
                yaw_quat(self.ref_motion["body_quat_w"][si, 0])[[1, 2, 3, 0]]
            ).inv()
        self.vr_3point_indices = [28,29,9] # by calling self.robot.find_bodies(["left_wrist_yaw_link",""right_wrist_yaw_link","torso_link"])
        self.ref_vr_3point_offsets = np.array([[0.18, -0.025, 0.0], [0.18,0.025, 0.0], [0.0,0.0,0.35]])  # relative to anchor point
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
        si = self.ref_motion_start_index
        return self.ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO]


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
        #print(control_signals["motion_anchor_pos_b"][2])
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
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()


class RLStreamingContactPolicy(RLBasePolicy):
    """Contact policy variant that consumes latest commands from Redis pub/sub."""

    def __init__(
        self,
        onnx_model_path,
        obs_names,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=[],
        hist_length=10,
        redis_ip=REDIS_IP,
        redis_port=REDIS_PORT,
        redis_channels=None,
        default_contact_label=None,
        use_8way_contact=False,
        use_10way_contact=False,
        use_5dim_contact_from_4dim=False,
        ref_motion_start_index=0,
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)
        self.ref_motion_start_index = max(0, int(ref_motion_start_index))
        self._limb_contact_file_is_8way = bool(use_8way_contact)
        self.default_value = {
            "q": np.array([float(x) for x in self.meta_data["default_joint_pos"].split(",")]),
            "dq": np.zeros(29),
        }
        self.action_scale = np.array([float(x) for x in self.meta_data["action_scale"].split(",")])
        if len(self.action_scale) != 29:
            self.action_scale = np.ones(29) * self.action_scale
        else:
            self.action_scale = self.action_scale[ISAAC_TO_MUJOCO]

        self.use_10way_contact = bool(use_10way_contact)
        self.use_5dim_contact_from_4dim = bool(use_5dim_contact_from_4dim)
        if self.use_5dim_contact_from_4dim and (not self.use_10way_contact) and use_8way_contact:
            raise ValueError(
                "use_5dim_contact_from_4dim (5-dim output) requires use_8way_contact: false, "
                "or enable use_10way_contact for 10-dim output with zero seat."
            )
        self.use_8way_contact = bool(use_8way_contact) or self.use_10way_contact
        if self.use_10way_contact:
            self.contact_dim = 10
        elif self.use_5dim_contact_from_4dim:
            self.contact_dim = 5
        else:
            self.contact_dim = 8 if use_8way_contact else 4
        self.motion_length = int(1e9)
        self.keyboard_controller = KeyboardController()
        self.lower_cmd_dim = 24 * lookahead_steps
        self.vr_pos_dim = 9
        self.vr_orn_dim = 12
        self.anchor_pos_dim = 3
        self.anchor_orn_dim = 4

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]
        lower_q_default = self.default_value["q"][self.lower_joint_indices].astype(np.float32)
        lower_dq_default = np.zeros_like(lower_q_default, dtype=np.float32)
        lower_cmd_single = np.hstack([lower_q_default, lower_dq_default]).astype(np.float32)
        self.latest_lower_cmd = np.tile(lower_cmd_single, lookahead_steps)

        self.vr_3point_offsets = np.array(
            [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
            dtype=np.float32,
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.fk_urdf_path = os.path.join(script_dir, "assets", "g1", "g1_29dof_kin_extended.urdf")
        self.vr_3point_link_names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        try:
            vr_pos_init, vr_orn_init = _compute_initial_vr_3point_local_from_pybullet(
                self.default_value["q"],
                self.fk_urdf_path,
                self.vr_3point_link_names,
                self.vr_3point_offsets,
                anchor_link_name="torso_link",
            )
            self.latest_vr_3point_pos = vr_pos_init
            self.latest_vr_3point_orn = vr_orn_init
        except Exception as exc:
            print(f"[RLStreamingContactPolicy] FK init failed ({exc}), falling back to zeros.", flush=True)
            self.latest_vr_3point_pos = np.zeros(self.vr_pos_dim, dtype=np.float32)
            self.latest_vr_3point_orn = np.zeros(self.vr_orn_dim, dtype=np.float32)
            self.latest_vr_3point_orn[0] = 1.0
            self.latest_vr_3point_orn[4] = 1.0
            self.latest_vr_3point_orn[8] = 1.0

        if default_contact_label is None:
            self.latest_contact_mask = np.zeros(self.contact_dim, dtype=np.float32)
        else:
            parsed_default_contact = np.asarray(default_contact_label, dtype=np.float32).reshape(-1)
            if self.use_10way_contact and self.use_5dim_contact_from_4dim:
                p = parsed_default_contact
                n = p.size
                if n == 4:
                    parsed_default_contact = expand_4way_limb_to_10way_with_zero_seat(
                        p.reshape(1, -1)
                    ).reshape(-1)
                elif n == 5:
                    parsed_default_contact = expand_4way_limb_to_10way_with_zero_seat(
                        p[:4].reshape(1, -1)
                    ).reshape(-1)
                elif n == 8:
                    if self._limb_contact_file_is_8way:
                        parsed_default_contact = np.hstack([p, np.zeros(2, dtype=np.float32)]).reshape(-1)
                    else:
                        m4 = reduce_8way_contact_to_4(p.reshape(1, -1)).reshape(-1)
                        parsed_default_contact = expand_4way_limb_to_10way_with_zero_seat(
                            m4.reshape(1, -1)
                        ).reshape(-1)
                elif n == 10:
                    parsed_default_contact = np.hstack([p[:8], np.zeros(2, dtype=np.float32)]).reshape(-1)
                else:
                    raise ValueError(
                        "default_contact_label: use_10way_contact + use_5dim_contact_from_4dim expects "
                        f"4, 5, 8, or 10 values, got {n}"
                    )
                if parsed_default_contact.size != self.contact_dim:
                    raise ValueError(
                        f"default_contact_label: expected {self.contact_dim} values after pad, "
                        f"got {parsed_default_contact.size}"
                    )
            elif self.use_10way_contact:
                if parsed_default_contact.size == 5:
                    parsed_default_contact = expand_5way_contact_to_10(
                        parsed_default_contact.reshape(1, -1)
                    ).reshape(-1)
                if parsed_default_contact.size != self.contact_dim:
                    raise ValueError(
                        "default_contact_label: use_10way_contact expects 5 values "
                        f"(4-way + pelvis_seat) or {self.contact_dim}, got {parsed_default_contact.size}"
                    )
            elif self.use_5dim_contact_from_4dim:
                if parsed_default_contact.size == 4:
                    parsed_default_contact = pad_4way_contact_to_5(
                        parsed_default_contact.reshape(1, -1)
                    ).reshape(-1)
                elif parsed_default_contact.size == 5:
                    parsed_default_contact = np.asarray(
                        np.hstack([parsed_default_contact[:4], 0.0]), dtype=np.float32
                    ).reshape(-1)
                if parsed_default_contact.size != self.contact_dim:
                    raise ValueError(
                        "default_contact_label: use_5dim_contact_from_4dim expects 4 values "
                        f"(padded to 5) or 5 (5th forced to 0), got {parsed_default_contact.size}"
                    )
            elif parsed_default_contact.size != self.contact_dim:
                raise ValueError(
                    f"default_contact_label must contain {self.contact_dim} values, got {parsed_default_contact.size}"
                )
            self.latest_contact_mask = parsed_default_contact.copy()
        self.latest_motion_anchor_pos_w = np.zeros(self.anchor_pos_dim, dtype=np.float32)
        self.latest_motion_anchor_orn_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        self.received_any_stream = False
        self._stream_msg_count = 0
        self._last_stream_warn_t = 0.0
        self._last_stream_info_t = 0.0

        if hist_length > 1:
            self.history_buffer = HistoryBuffer(hist_length, obs_names=hist_names, flatten=True)

        if redis_channels is None:
            redis_channels = {
                "lower_cmd": "lower_cmd",
                "vr_3point_pos_l": "vr_3point_pos_l",
                "vr_3point_orn_l": "vr_3point_orn_l",
                "contact_mask": "contact_mask",
                "motion_anchor_pos_w": "motion_anchor_pos_w",
                "motion_anchor_orn_w": "motion_anchor_orn_w",
            }
        self.redis_channels = redis_channels
        self.redis_client = redis.Redis(host=redis_ip, port=redis_port, db=0)
        self.pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(
            self.redis_channels["lower_cmd"],
            self.redis_channels["vr_3point_pos_l"],
            self.redis_channels["vr_3point_orn_l"],
            self.redis_channels["contact_mask"],
            self.redis_channels["motion_anchor_pos_w"],
            self.redis_channels["motion_anchor_orn_w"],
        )
        self._channel_to_key = {
            self.redis_channels["lower_cmd"]: "lower_cmd",
            self.redis_channels["vr_3point_pos_l"]: "vr_3point_pos_l",
            self.redis_channels["vr_3point_orn_l"]: "vr_3point_orn_l",
            self.redis_channels["contact_mask"]: "contact_mask",
            self.redis_channels["motion_anchor_pos_w"]: "motion_anchor_pos_w",
            self.redis_channels["motion_anchor_orn_w"]: "motion_anchor_orn_w",
        }
        print(
            f"[RLStreamingContactPolicy] contact_dim={self.contact_dim} "
            f"use_8way_contact={use_8way_contact} use_10way_contact={self.use_10way_contact} "
            f"use_5dim_contact_from_4dim={self.use_5dim_contact_from_4dim} — Listening Redis channels:"
            f" {self.redis_channels['lower_cmd']}, {self.redis_channels['vr_3point_pos_l']},"
            f" {self.redis_channels['vr_3point_orn_l']}, {self.redis_channels['contact_mask']},"
            f" {self.redis_channels['motion_anchor_pos_w']}, {self.redis_channels['motion_anchor_orn_w']}"
        )

    def get_q_init(self):
        return self.default_value["q"][ISAAC_TO_MUJOCO].copy()

    @staticmethod
    def _decode_array(payload, expected_dim, allow_single_tile_for_lower=False):
        arr = pickle.loads(payload)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if allow_single_tile_for_lower and expected_dim % 24 == 0 and arr.size == 24 and expected_dim > 24:
            arr = np.tile(arr, expected_dim // 24)
        if arr.size != expected_dim:
            raise ValueError(f"Expected payload size {expected_dim}, got {arr.size}")
        return arr

    def _pull_latest_commands(self):
        while True:
            msg = self.pubsub.get_message(timeout=0.0)
            if msg is None:
                break
            if msg.get("type") != "message":
                continue
            channel = msg["channel"]
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            data = msg["data"]
            if not isinstance(data, (bytes, bytearray)):
                continue
            try:
                key = self._channel_to_key.get(channel)
                if key == "lower_cmd":
                    self.latest_lower_cmd = self._decode_array(
                        data,
                        self.lower_cmd_dim,
                        allow_single_tile_for_lower=True,
                    )
                elif key == "vr_3point_pos_l":
                    self.latest_vr_3point_pos = self._decode_array(data, self.vr_pos_dim)
                elif key == "vr_3point_orn_l":
                    self.latest_vr_3point_orn = self._decode_array(data, self.vr_orn_dim)
                elif key == "contact_mask":
                    arr = pickle.loads(data)
                    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                    if self.use_10way_contact and self.use_5dim_contact_from_4dim:
                        n = arr.size
                        if n == 4:
                            arr = expand_4way_limb_to_10way_with_zero_seat(arr.reshape(1, -1)).reshape(-1)
                        elif n == 5:
                            arr = expand_4way_limb_to_10way_with_zero_seat(arr[:4].reshape(1, -1)).reshape(-1)
                        elif n == 8:
                            if self._limb_contact_file_is_8way:
                                arr = np.hstack([arr, np.zeros(2, dtype=np.float32)])
                            else:
                                m4 = reduce_8way_contact_to_4(arr.reshape(1, -1)).reshape(-1)
                                arr = expand_4way_limb_to_10way_with_zero_seat(m4.reshape(1, -1)).reshape(-1)
                        elif n == 10:
                            arr = np.hstack([arr[:8], np.zeros(2, dtype=np.float32)])
                        else:
                            raise ValueError(
                                f"contact stream: use_10way + use_5dim_from_4dim expected 4,5,8,10 values, got {n}"
                            )
                        if arr.size != self.contact_dim:
                            raise ValueError(
                                f"contact stream: expected {self.contact_dim} values after pad, got {arr.size}"
                            )
                    elif self.use_10way_contact:
                        if arr.size == 5:
                            arr = expand_5way_contact_to_10(arr.reshape(1, -1)).reshape(-1)
                        if arr.size != self.contact_dim:
                            raise ValueError(
                                f"contact stream: expected 5 or {self.contact_dim} values, got {arr.size}"
                            )
                    elif self.use_5dim_contact_from_4dim:
                        if arr.size == 4:
                            arr = pad_4way_contact_to_5(arr.reshape(1, -1)).reshape(-1)
                        elif arr.size == 5:
                            arr = arr.copy()
                            arr[4] = 0.0
                        if arr.size != self.contact_dim:
                            raise ValueError(
                                f"contact stream: expected 4 or {self.contact_dim} values, got {arr.size}"
                            )
                    elif arr.size != self.contact_dim:
                        raise ValueError(f"Expected payload size {self.contact_dim}, got {arr.size}")
                    self.latest_contact_mask = arr
                elif key == "motion_anchor_pos_w":
                    self.latest_motion_anchor_pos_w = self._decode_array(data, self.anchor_pos_dim)
                elif key == "motion_anchor_orn_w":
                    self.latest_motion_anchor_orn_w = self._decode_array(data, self.anchor_orn_dim)
                self.received_any_stream = True
                self._stream_msg_count += 1
            except Exception as exc:
                now = time.time()
                if now - self._last_stream_warn_t > 2.0:
                    print(f"[RLStreamingContactPolicy] Stream decode warning on channel '{channel}': {exc}", flush=True)
                    self._last_stream_warn_t = now
                continue

    def prepare_control_signals(self, robot_state):
        self._pull_latest_commands()
        now = time.time()
        if self.received_any_stream:
            if now - self._last_stream_info_t > 2.0:
                print(f"[RLStreamingContactPolicy] stream alive: messages={self._stream_msg_count}", flush=True)
                self._last_stream_info_t = now
        else:
            if now - self._last_stream_warn_t > 2.0:
                print("[RLStreamingContactPolicy] waiting for stream packets...", flush=True)
                self._last_stream_warn_t = now
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        target_anchor_rot = Rotation.from_quat(self.latest_motion_anchor_orn_w)
        error = self.latest_motion_anchor_pos_w - robot_state.root_pos
        #error[2] = 0.0
        control_signals = {
            "lower_command": self.latest_lower_cmd.copy(),
            "vr_3point_pos": self.latest_vr_3point_pos.copy(),
            "vr_3point_ori": self.latest_vr_3point_orn.copy(),
            "contact_mask": self.latest_contact_mask.copy(),
            "compliance": self.keyboard_controller.get_compliance(),
            "motion_anchor_pos_b": anchor_rot_inv.apply(error).astype(np.float32),
            "motion_anchor_ori_b": (anchor_rot_inv * target_anchor_rot).as_matrix()[:, :2].reshape(-1).astype(np.float32),
        }
        control_signals["projected_gravity"] = anchor_rot_inv.apply(np.array([0, 0, -1]))
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
        return np.concatenate(obs).reshape(1, -1)

    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape, f"Obs shape: {obs.shape}, input shape: {self.input_shape}"
        ort_inputs = {"obs": obs.astype(np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return ort_outs[0].flatten()