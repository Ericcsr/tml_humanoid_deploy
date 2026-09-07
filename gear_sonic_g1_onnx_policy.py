"""
GEAR-SONIC / MuJoCo deploy: SONIC universal-token policy in Python.

Matches ``gear_sonic_deploy`` (C++): separate **encoder** and **decoder** ONNX models
under ``exported_policies/gear_sonic/`` (git-lfs) — same split as ``gear_sonic/eval_agent_trl.py``
with ``export_onnx_only=True`` (``*_encoder.onnx`` + ``*_decoder.onnx`` from
``inference_helpers.export_universal_token_encoders_as_onnx`` and
``export_universal_token_decoder_as_onnx``).

Encoder I/O (PyTorch export): input ``obs_dict`` → flat ``[encoder_index | tokenizer…]``;
output ``encoded_tokens``. Decoder: input ``[encoded_tokens | tokenizer… | proprio]``;
for ``g1_dyn`` the tokenizer tail is often empty so input is ``[tokens | proprio]``.

Optional **fused** single ONNX (``*_g1.onnx`` from ``export_universal_token_module_as_onnx``):
set ``onnx_model_path`` only and omit encoder/decoder paths.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime
from scipy.spatial.transform import Rotation

from rl_policy import clamp_ref_motion_start_index
from utils.math_utils import heading_zup, yaw_quat_xyzw
from utils.params import ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC


def _default_gear_sonic_release_dir() -> Path:
    """Split encoder/decoder ONNX live under ``exported_policies/gear_sonic/`` (git-lfs)."""
    return Path(__file__).resolve().parent / "exported_policies" / "gear_sonic"


def _ort_providers(preferred: Sequence[str] | None) -> list[str]:
    if preferred is not None:
        pref = list(preferred)
    else:
        pref = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    avail = set(onnxruntime.get_available_providers())
    out = [p for p in pref if p in avail]
    if not out:
        out = ["CPUExecutionProvider"]
    return out


def _onnx_last_dim(shape) -> int:
    """Product of static dimensions; -1 if any axis is symbolic."""
    d = 1
    for x in shape:
        if x is None or x == "None":
            return -1
        d *= int(x)
    return d


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _rot6d_robot_to_ref(robot_quat_xyzw: np.ndarray, ref_quat_xyzw: np.ndarray) -> np.ndarray:
    r_r = Rotation.from_quat(np.asarray(robot_quat_xyzw, dtype=np.float64).reshape(4))
    r_ref = Rotation.from_quat(np.asarray(ref_quat_xyzw, dtype=np.float64).reshape(4))
    m = (r_r.inv() * r_ref).as_matrix()
    return m[:, :2].reshape(-1).astype(np.float32)


def _yaw_rotation(quat_xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(yaw_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)))


def _motion_anchor_pos_b_world(
    robot_pos: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    ref_pos: np.ndarray,
) -> np.ndarray:
    r_r = Rotation.from_quat(np.asarray(robot_quat_xyzw, dtype=np.float64).reshape(4))
    return r_r.inv().apply(np.asarray(ref_pos, dtype=np.float64) - np.asarray(robot_pos, dtype=np.float64)).astype(
        np.float32
    )


def _motion_anchor_ori_b_world(
    robot_pos: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    ref_pos: np.ndarray,
    ref_quat_xyzw: np.ndarray,
) -> np.ndarray:
    r_r = Rotation.from_quat(np.asarray(robot_quat_xyzw, dtype=np.float64).reshape(4))
    r_ref = Rotation.from_quat(np.asarray(ref_quat_xyzw, dtype=np.float64).reshape(4))
    _, ori = _isaac_subtract_frame_transforms(
        np.asarray(robot_pos, dtype=np.float64),
        r_r.as_quat(),
        np.asarray(ref_pos, dtype=np.float64),
        r_ref.as_quat(),
    )
    m = Rotation.from_quat(ori).as_matrix()
    return m[:, :2].reshape(-1).astype(np.float32)


def _isaac_subtract_frame_transforms(
    pos_a: np.ndarray, quat_a_xyzw: np.ndarray, pos_b: np.ndarray, quat_b_xyzw: np.ndarray
):
    r_a = Rotation.from_quat(quat_a_xyzw)
    r_b = Rotation.from_quat(quat_b_xyzw)
    pos_rel = r_a.inv().apply(pos_b - pos_a)
    quat_rel = (r_a.inv() * r_b).as_quat()
    return pos_rel, quat_rel


DEFAULT_TOKENIZER_ORDER: tuple[str, ...] = (
    "command_multi_future_nonflat",
    "motion_anchor_ori_b_mf_nonflat",
    "command_z_multi_future_nonflat",
)

DEPLOY_ENCODER_LAYOUT: tuple[tuple[str, int], ...] = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)

DEPLOY_DECODER_DIM = 994
DEPLOY_ENCODER_DIM = sum(dim for _, dim in DEPLOY_ENCODER_LAYOUT)
LOWER_BODY_MUJOCO_ORDER_IN_ISAAC = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
WRIST_ISAAC_ORDER_IN_ISAAC = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)
GEAR_SONIC_DEFAULT_Q_MUJOCO = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
GEAR_SONIC_ACTION_SCALE_MUJOCO = np.array(
    [
        0.3506614663788243,
        0.3506614663788243,
        0.5475464652142303,
        0.3506614663788243,
        0.43857731392336724,
        0.43857731392336724,
        0.3506614663788243,
        0.3506614663788243,
        0.5475464652142303,
        0.3506614663788243,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464652142303,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
    ],
    dtype=np.float64,
)


class RLGearSonicG1OnnxPolicy:
    """
    Whole-body tracking: SONIC G1 motion tokenizer + ``g1_dyn`` decoder (split or fused ONNX).
    """

    def __init__(
        self,
        ref_motion_path: str,
        *,
        fused_onnx_path: str | None = None,
        encoder_onnx_path: str | None = None,
        decoder_onnx_path: str | None = None,
        num_future_frames: int = 5,
        lookahead_frame_skips: int = 1,
        init_at_first_frame: bool = False,
        ref_motion_start_index: int = 0,
        num_dof: int = 29,
        tokenizer_feature_order: Sequence[str] | None = None,
        default_joint_pos: np.ndarray | None = None,
        action_scale: np.ndarray | None = None,
        onnx_providers: Sequence[str] | tuple[str, ...] | None = None,
        encoder_index: float = 0.0,
        encoder_tokenizer_slice: tuple[int, int] | None = None,
        omega_is_world_frame: bool = False,
        apply_heading_alignment: bool = False,
        use_history_buffer: bool = True,
        use_root_state: bool = False,
    ):
        prov = _ort_providers(onnx_providers)
        self._split = fused_onnx_path is None or str(fused_onnx_path).strip() == ""
        self.encoder_index = float(encoder_index)
        self._encoder_tokenizer_slice = encoder_tokenizer_slice
        self.omega_is_world_frame = bool(omega_is_world_frame)
        self.apply_heading_alignment = bool(apply_heading_alignment)
        self.use_history_buffer = bool(use_history_buffer)
        # False: ignore privileged/odom root pos and linear velocity (IMU orientation + joints only).
        self.use_root_state = bool(use_root_state)
        if not self.use_root_state:
            print(
                "[RLGearSonicG1OnnxPolicy] use_root_state=False: IMU orientation + joints only "
                "(root pos / linear vel unused).",
                flush=True,
            )

        self.num_future_frames = int(num_future_frames)
        self.lookahead_frame_skips = int(lookahead_frame_skips)
        self.num_dof = int(num_dof)
        self.tokenizer_feature_order = tuple(
            tokenizer_feature_order if tokenizer_feature_order is not None else DEFAULT_TOKENIZER_ORDER
        )

        if self._split:
            if not encoder_onnx_path or not decoder_onnx_path:
                raise ValueError("Split mode requires encoder_onnx_path and decoder_onnx_path (or use fused_onnx_path).")
            self._enc_session = onnxruntime.InferenceSession(encoder_onnx_path, providers=prov)
            self._dec_session = onnxruntime.InferenceSession(decoder_onnx_path, providers=prov)
            self._enc_in = self._enc_session.get_inputs()[0].name
            self._dec_in = self._dec_session.get_inputs()[0].name
            enc_shape = tuple(self._enc_session.get_inputs()[0].shape)
            self._encoder_in_dim = _onnx_last_dim(enc_shape)
            if self._encoder_in_dim < 0:
                raise ValueError("Encoder ONNX input has symbolic shape; need fixed batch and feature dim.")
            tok_shape = tuple(self._enc_session.get_outputs()[0].shape)
            self._token_dim = _onnx_last_dim(tok_shape)
            if self._token_dim < 0:
                z = np.zeros((1, self._encoder_in_dim), dtype=np.float32)
                y = self._enc_session.run(None, {self._enc_in: z})[0]
                self._token_dim = int(np.asarray(y).size)
            dec_shape = tuple(self._dec_session.get_inputs()[0].shape)
            self._decoder_in_dim = _onnx_last_dim(dec_shape)
            self.session = None
            self._in_name = None
        else:
            self.session = onnxruntime.InferenceSession(str(fused_onnx_path), providers=prov)
            self._in_name = self.session.get_inputs()[0].name
            self._enc_session = None
            self._dec_session = None
            self._encoder_in_dim = -1
            self._decoder_in_dim = -1
            self._token_dim = -1

        self.input_shape = tuple(self.session.get_inputs()[0].shape) if self.session else (-1, -1)
        self.expected_obs_dim = (
            int(self.input_shape[-1]) if (self.session and self.input_shape[-1] not in (None, "None")) else -1
        )

        meta = self.session.get_modelmeta().custom_metadata_map if self.session else {}
        if self._split and self._dec_session is not None:
            meta = {**self._dec_session.get_modelmeta().custom_metadata_map, **meta}
        if default_joint_pos is not None:
            self.default_value = {
                "q": np.asarray(default_joint_pos, dtype=np.float64).reshape(-1),
                "dq": np.zeros(self.num_dof, dtype=np.float64),
            }
        elif "default_joint_pos" in meta:
            self.default_value = {
                "q": np.array([float(x) for x in meta["default_joint_pos"].split(",")], dtype=np.float64),
                "dq": np.zeros(self.num_dof, dtype=np.float64),
            }
        else:
            # Split release ONNX omits metadata; match gear_sonic_deploy::default_angles.
            self.default_value = {
                "q": GEAR_SONIC_DEFAULT_Q_MUJOCO[MUJOCO_TO_ISAAC].copy(),
                "dq": np.zeros(self.num_dof, dtype=np.float64),
            }
            if self.default_value["q"].size != self.num_dof:
                raise ValueError(
                    f"gear_sonic default_angles Isaac view length {self.default_value['q'].size} != num_dof {self.num_dof}"
                )
            print(
                "[RLGearSonicG1OnnxPolicy] No default_joint_pos in ONNX metadata; using "
                "gear_sonic_deploy default_angles. Override with gear_sonic_default_joint_pos if needed.",
                flush=True,
            )

        action_scale_in_mujoco_order = False
        if action_scale is not None:
            asc = np.asarray(action_scale, dtype=np.float64).reshape(-1)
        elif "action_scale" in meta:
            asc = np.array([float(x) for x in meta["action_scale"].split(",")], dtype=np.float64)
        else:
            asc = GEAR_SONIC_ACTION_SCALE_MUJOCO.copy()
            if asc.size != self.num_dof:
                raise ValueError(
                    f"GEAR_SONIC_ACTION_SCALE_MUJOCO length {asc.size} != num_dof {self.num_dof}; "
                    "set gear_sonic_action_scale in YAML."
                )
            action_scale_in_mujoco_order = True
            print(
                "[RLGearSonicG1OnnxPolicy] No action_scale in ONNX metadata; using "
                "gear_sonic_deploy g1_action_scale (Mujoco order). Override with gear_sonic_action_scale if needed.",
                flush=True,
            )
        if asc.size == 1:
            asc = np.ones(self.num_dof, dtype=np.float64) * asc[0]
        elif asc.size == self.num_dof:
            if not action_scale_in_mujoco_order:
                asc = asc[ISAAC_TO_MUJOCO]
        else:
            asc = np.ones(self.num_dof, dtype=np.float64) * float(asc.reshape(-1)[0])
        self.action_scale = asc.astype(np.float64)
        self.default_q_mujoco = self.default_value["q"][ISAAC_TO_MUJOCO].copy()

        self.ref_motion = np.load(ref_motion_path)
        self.motion_length = int(self.ref_motion["joint_pos"].shape[0])
        self.ref_motion_start_index = clamp_ref_motion_start_index(ref_motion_start_index, self.motion_length)
        self.ticker = self.ref_motion_start_index
        self.init_at_first_frame = bool(init_at_first_frame)
        if self.init_at_first_frame:
            self.init_root_pos = np.zeros(3, dtype=np.float64)
            self.init_root_heading_inv = Rotation.identity()
        else:
            si = self.ref_motion_start_index
            self.init_root_pos = self.ref_motion["body_pos_w"][si, 0].copy().astype(np.float64)
            self.init_root_pos[2] = 0.0
            ref_q_wxyz = self.ref_motion["body_quat_w"][si, 0].astype(np.float64)
            ref_q_xyzw = _wxyz_to_xyzw(ref_q_wxyz)
            self.init_root_heading_inv = Rotation.from_quat(yaw_quat_xyzw(ref_q_xyzw)).inv()

        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:, 0].copy()
        self.ref_anchor_orns_wxyz = self.ref_motion["body_quat_w"][:, 0].copy()
        # xyzw alias used by run_mujoco_eval metrics / EE error helpers.
        self.ref_anchor_orns = self.ref_anchor_orns_wxyz[:, [1, 2, 3, 0]].copy()
        self._heading_initialized = False
        self._apply_delta_heading = Rotation.identity()
        self._history = deque(maxlen=10)

        self._proprio_dim = 2 * self.num_dof + 3 + 6 + 3 + 3 + self.num_dof + self.num_dof + self.num_dof
        tok_dim = self._tokenizer_flat_dim()

        if self._split:
            if self._decoder_in_dim > 0 and self._decoder_in_dim not in (
                self._token_dim + self._proprio_dim,
                DEPLOY_DECODER_DIM,
            ):
                print(
                    f"[RLGearSonicG1OnnxPolicy] Warning: decoder input dim {self._decoder_in_dim} != "
                    f"minimal dim {self._token_dim + self._proprio_dim} or deploy dim {DEPLOY_DECODER_DIM}.",
                    flush=True,
                )
            self._validate_encoder_layout(tok_dim)
        elif self.expected_obs_dim > 0:
            if self.expected_obs_dim != tok_dim + self._proprio_dim:
                print(
                    f"[RLGearSonicG1OnnxPolicy] Warning: fused ONNX dim {self.expected_obs_dim} != "
                    f"tokenizer({tok_dim})+proprio({self._proprio_dim}).",
                    flush=True,
                )

    def _validate_encoder_layout(self, tok_dim: int) -> None:
        if self._encoder_in_dim <= 0:
            return
        if self._encoder_tokenizer_slice is not None:
            s, e = self._encoder_tokenizer_slice
            if e - s != tok_dim:
                raise ValueError(
                    f"gear_sonic_encoder_tokenizer_slice [{s}, {e}) length {e - s} != G1 tokenizer dim {tok_dim}"
                )
            if s < 0 or e > self._encoder_in_dim or s >= e:
                raise ValueError(f"Invalid gear_sonic_encoder_tokenizer_slice [{s}, {e}) for encoder dim {self._encoder_in_dim}")
            if s != 0 and s != 1:
                print(
                    "[RLGearSonicG1OnnxPolicy] Note: encoder_index is assumed at column 0; tokenizer slice should align.",
                    flush=True,
                )
            return
        if self._encoder_in_dim == 1 + tok_dim:
            return
        if self._encoder_in_dim == DEPLOY_ENCODER_DIM:
            return
        raise ValueError(
            f"Encoder ONNX input dim is {self._encoder_in_dim} but G1 tokenizer is {tok_dim} (expected {1 + tok_dim} "
            f"for [encoder_index|tokenizer] or {DEPLOY_ENCODER_DIM} for gear_sonic_deploy observation_config). "
            f"Set gear_sonic_encoder_tokenizer_slice: [start, end) in YAML to paste the {tok_dim}-dim tokenizer "
            "into a custom flat encoder buffer, or use a fused *_g1.onnx."
        )

    def _frame_index(self, offset: int) -> int:
        idx = self.ticker + offset * self.lookahead_frame_skips
        if idx >= self.motion_length:
            idx = self.motion_length - 1
        if idx < 0:
            idx = 0
        return int(idx)

    def _tokenizer_flat_dim(self) -> int:
        h = self.num_future_frames
        d = self.num_dof
        parts = {
            "command_multi_future_nonflat": 2 * h * d,
            "motion_anchor_ori_b_mf_nonflat": 6 * h,
            "command_z_multi_future_nonflat": h,
        }
        return sum(parts[name] for name in self.tokenizer_feature_order)

    def _build_command_multi_future_nonflat(self) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for i in range(self.num_future_frames):
            fi = self._frame_index(i)
            jp = self.ref_q_pos[fi].astype(np.float32)
            jv = self.ref_q_vel[fi].astype(np.float32)
            chunks.append(np.concatenate([jp, jv]).astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def _build_motion_anchor_ori_b_mf_nonflat(self, robot_state) -> np.ndarray:
        r_quat = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        parts: List[np.ndarray] = []
        for i in range(self.num_future_frames):
            fi = self._frame_index(i)
            ref_q = self._corrected_ref_quat_xyzw(fi)
            parts.append(_rot6d_robot_to_ref(r_quat, ref_q))
        return np.concatenate(parts, axis=0)

    def _build_command_z_multi_future_nonflat(self) -> np.ndarray:
        zs = []
        for i in range(self.num_future_frames):
            fi = self._frame_index(i)
            zs.append(np.float32(self.ref_anchor_poses[fi, 2]))
        return np.stack(zs, axis=0)

    def _build_tokenizer_flat(self, robot_state) -> np.ndarray:
        builders = {
            "command_multi_future_nonflat": lambda: self._build_command_multi_future_nonflat(),
            "motion_anchor_ori_b_mf_nonflat": lambda: self._build_motion_anchor_ori_b_mf_nonflat(robot_state),
            "command_z_multi_future_nonflat": lambda: self._build_command_z_multi_future_nonflat(),
        }
        return np.concatenate([builders[k]() for k in self.tokenizer_feature_order], axis=0).astype(np.float32)

    def _build_encoder_input(self, tok_flat: np.ndarray) -> np.ndarray:
        if self._encoder_in_dim == DEPLOY_ENCODER_DIM and self._encoder_tokenizer_slice is None:
            return self._build_deploy_encoder_input()
        buf = np.zeros((1, self._encoder_in_dim), dtype=np.float32)
        buf[0, 0] = np.float32(self.encoder_index)
        t = np.asarray(tok_flat, dtype=np.float32).reshape(-1)
        if self._encoder_tokenizer_slice is not None:
            s, e = self._encoder_tokenizer_slice
            buf[0, s:e] = t
        else:
            buf[0, 1 : 1 + t.size] = t
        return buf

    def _motion_joint_frames(self, num_frames: int, step: int, indexes: np.ndarray | None = None, vel: bool = False) -> np.ndarray:
        src = self.ref_q_vel if vel else self.ref_q_pos
        frames = []
        for i in range(num_frames):
            fi = self._frame_index(i * step)
            vals = src[fi].astype(np.float32)
            if indexes is not None:
                vals = vals[indexes]
            frames.append(vals.reshape(-1))
        return np.concatenate(frames, axis=0).astype(np.float32)

    def _motion_root_z_frames(self, num_frames: int, step: int) -> np.ndarray:
        return np.array([self.ref_anchor_poses[self._frame_index(i * step), 2] for i in range(num_frames)], dtype=np.float32)

    def _ensure_heading_alignment(self, robot_state) -> None:
        if self._heading_initialized:
            return
        if not self.apply_heading_alignment:
            self._apply_delta_heading = Rotation.identity()
            self._heading_initialized = True
            return
        robot_q = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        ref_q0 = _wxyz_to_xyzw(self.ref_anchor_orns_wxyz[self._frame_index(0)])
        # gear_sonic_deploy ComputeApplyDeltaHeading:
        #   apply_delta_heading = heading(init_base_quat) * inv_heading(init_ref_root_quat)
        self._apply_delta_heading = _yaw_rotation(robot_q) * _yaw_rotation(ref_q0).inv()
        self._heading_initialized = True
        robot_yaw = heading_zup(robot_q)
        ref_yaw = heading_zup(ref_q0)
        print(
            f"[RLGearSonicG1OnnxPolicy] Heading alignment: robot_yaw={np.degrees(robot_yaw):.1f}° "
            f"ref_yaw={np.degrees(ref_yaw):.1f}° Δ={np.degrees(robot_yaw - ref_yaw):.1f}°",
            flush=True,
        )

    def _corrected_ref_quat_xyzw(self, frame_idx: int) -> np.ndarray:
        ref_q = _wxyz_to_xyzw(self.ref_anchor_orns_wxyz[frame_idx])
        return (self._apply_delta_heading * Rotation.from_quat(ref_q)).as_quat()

    def _motion_anchor_ori_frames(self, robot_state, num_frames: int, step: int) -> np.ndarray:
        r_quat = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        parts = []
        for i in range(num_frames):
            fi = self._frame_index(i * step)
            parts.append(_rot6d_robot_to_ref(r_quat, self._corrected_ref_quat_xyzw(fi)))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _build_deploy_encoder_input(self) -> np.ndarray:
        # Matches gear_sonic_deploy/policy/release/observation_config.yaml encoder_observations order.
        buf = np.zeros((1, DEPLOY_ENCODER_DIM), dtype=np.float32)
        offset = 0
        values = {
            "encoder_mode_4": np.array([self.encoder_index, 0.0, 0.0, 0.0], dtype=np.float32),
            "motion_joint_positions_10frame_step5": self._motion_joint_frames(10, 5),
            "motion_joint_velocities_10frame_step5": self._motion_joint_frames(10, 5, vel=True),
            "motion_root_z_position_10frame_step5": self._motion_root_z_frames(10, 5),
            "motion_root_z_position": self._motion_root_z_frames(1, 1),
            "motion_anchor_orientation": self._motion_anchor_ori_frames(self._latest_robot_state, 1, 1),
            "motion_anchor_orientation_10frame_step5": self._motion_anchor_ori_frames(self._latest_robot_state, 10, 5),
            "motion_joint_positions_lowerbody_10frame_step5": self._motion_joint_frames(10, 5, LOWER_BODY_MUJOCO_ORDER_IN_ISAAC),
            "motion_joint_velocities_lowerbody_10frame_step5": self._motion_joint_frames(10, 5, LOWER_BODY_MUJOCO_ORDER_IN_ISAAC, vel=True),
            "motion_joint_positions_wrists_10frame_step1": self._motion_joint_frames(10, 1, WRIST_ISAAC_ORDER_IN_ISAAC),
        }
        for name, dim in DEPLOY_ENCODER_LAYOUT:
            arr = values.get(name)
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                if arr.size != dim:
                    raise ValueError(f"Deploy encoder observation {name} expected {dim} values, got {arr.size}")
                buf[0, offset : offset + dim] = arr
            offset += dim
        return buf

    def _base_angular_velocity_body(self, robot_state) -> np.ndarray:
        """Match ``g1_deploy_onnx_ref``: gyro is in the **base / IMU** frame.

        MuJoCo free-joint ``qvel[3:6]`` is angular velocity in the **body frame** — confirmed by
        integration test: at 90° yaw, ``qvel=[1,0,0]`` rotates around world +y (= body +x), not
        world +x.  Therefore ``omega_is_world_frame`` should be ``False`` for both MuJoCo
        simulation and real hardware (IMU gyroscope also outputs body-frame angular rate).

        Set ``omega_is_world_frame=True`` only if your driver explicitly provides world-frame ω.
        """
        robot_q = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        omega_raw = np.asarray(robot_state.omega, dtype=np.float64).reshape(3)
        if self.omega_is_world_frame:
            return Rotation.from_quat(robot_q).inv().apply(omega_raw).astype(np.float32)
        return omega_raw.astype(np.float32)

    def _proprio_flat(self, robot_state, mid: int) -> np.ndarray:
        ref_q = self._corrected_ref_quat_xyzw(mid)
        robot_q = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        cmd = np.concatenate(
            [
                self.ref_q_pos[mid].astype(np.float32),
                self.ref_q_vel[mid].astype(np.float32),
            ]
        )
        if self.use_root_state:
            ref_p = self.ref_anchor_poses[mid].astype(np.float64)
            robot_root = np.asarray(robot_state.root_pos, dtype=np.float64).reshape(3)
            map_b = _motion_anchor_pos_b_world(robot_root, robot_q, ref_p)
            o_b = _motion_anchor_ori_b_world(robot_root, robot_q, ref_p, ref_q)
            root_vel_w = np.asarray(robot_state.root_vel, dtype=np.float64).reshape(3)
            r_body = Rotation.from_quat(robot_q)
            base_lin_b = r_body.inv().apply(root_vel_w).astype(np.float32)
        else:
            # Match C++ deploy: no estimated/privileged root xy or linear velocity.
            map_b = np.zeros(3, dtype=np.float32)
            o_b = _rot6d_robot_to_ref(robot_q, ref_q)
            base_lin_b = np.zeros(3, dtype=np.float32)
        base_ang_b = self._base_angular_velocity_body(robot_state)
        q_isaac = np.asarray(robot_state.q, dtype=np.float64)[MUJOCO_TO_ISAAC]
        dq_isaac = np.asarray(robot_state.dq, dtype=np.float64)[MUJOCO_TO_ISAAC]
        q_def = self.default_value["q"]
        joint_pos_rel = (q_isaac - q_def).astype(np.float32)
        joint_vel = dq_isaac.astype(np.float32)
        last_act = np.asarray(robot_state.last_action, dtype=np.float32).reshape(-1)
        return np.concatenate(
            [cmd, map_b.astype(np.float32), o_b, base_lin_b.reshape(-1), base_ang_b, joint_pos_rel, joint_vel, last_act]
        ).astype(np.float32)

    def _current_deploy_history_state(self, robot_state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        robot_q = np.asarray(robot_state.root_orn, dtype=np.float64).reshape(4)
        omega = self._base_angular_velocity_body(robot_state)
        # C++ logger uses body_q[i] = motor_state[mujoco_to_isaaclab[i]] - default[mujoco_to_isaaclab[i]].
        # With robot_state.q in MuJoCo order, this produces the policy/IsaacLab joint order.
        q_mujoco_rel = (
            np.asarray(robot_state.q, dtype=np.float64).reshape(-1)[MUJOCO_TO_ISAAC]
            - self.default_q_mujoco[MUJOCO_TO_ISAAC]
        ).astype(np.float32)
        dq_mujoco = np.asarray(robot_state.dq, dtype=np.float32).reshape(-1)[MUJOCO_TO_ISAAC]
        last_action = np.asarray(robot_state.last_action, dtype=np.float32).reshape(-1)
        gravity = Rotation.from_quat(robot_q).inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)
        return omega, q_mujoco_rel, dq_mujoco, last_action, gravity

    def _deploy_history_state(self, robot_state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """10-frame history matching ``StateLogger::GetLatest(10, control_dt, newest_first=false)``.

        ``state_logger.cpp`` (stride branch): collect **newest → oldest** into ``out``, append
        ``makeZeroEntry`` until ``len(out)==n``, then ``reverse`` so the flat buffer is
        **oldest → newest**. Missing past timesteps become **leading** zeros; the **current**
        timestep is always the **last** 3+29+… slice (index 9 in each 10-block). Padding zeros
        after the newest in the *unreversed* list becomes leading zeros after reverse — not
        trailing zeros after the real samples.
        """
        cur = self._current_deploy_history_state(robot_state)
        if not self.use_history_buffer:
            return tuple(np.tile(x, 10) for x in cur)
        self._history.append(cur)
        zero_entry = (
            np.zeros(3, dtype=np.float32),
            np.zeros(self.num_dof, dtype=np.float32),
            np.zeros(self.num_dof, dtype=np.float32),
            np.zeros(self.num_dof, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        hist = list(self._history)
        # Newest-first with stride 1, same indexing as ring walk from newest_idx backward.
        newest_first: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for j in range(10):
            if j >= len(hist):
                break
            newest_first.append(hist[-1 - j])
        while len(newest_first) < 10:
            newest_first.append(zero_entry)
        entries = list(reversed(newest_first))
        return (
            np.concatenate([e[0] for e in entries], axis=0),
            np.concatenate([e[1] for e in entries], axis=0),
            np.concatenate([e[2] for e in entries], axis=0),
            np.concatenate([e[3] for e in entries], axis=0),
            np.concatenate([e[4] for e in entries], axis=0),
        )

    def _build_deploy_decoder_input(self, token: np.ndarray, robot_state) -> np.ndarray:
        # Layout from gear_sonic_deploy/policy/release/observation_config.yaml:
        # token_state(64), his_base_ang_vel(30), q(290), dq(290), last_action(290), gravity(30).
        token = np.asarray(token, dtype=np.float32).reshape(-1)
        omega, q_rel, dq, last_action, gravity = self._deploy_history_state(robot_state)
        if token.size != self._token_dim:
            raise ValueError(f"Encoder token dim mismatch: expected {self._token_dim}, got {token.size}")
        parts = [token, omega, q_rel, dq, last_action, gravity]
        obs = np.concatenate(parts, axis=0).astype(np.float32).reshape(1, -1)
        if obs.shape[-1] != DEPLOY_DECODER_DIM:
            raise ValueError(f"Deploy decoder observation expected {DEPLOY_DECODER_DIM}, got {obs.shape[-1]}")
        return obs

    def get_q_init(self):
        si = self.ref_motion_start_index
        return self.ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO].copy()

    def prepare_control_signals(self, robot_state):
        return {}

    def prepare_obs(self, robot_state, control_signals):
        del control_signals
        self._latest_robot_state = robot_state
        self._ensure_heading_alignment(robot_state)
        mid = self.ticker if self.ticker < self.motion_length else self.motion_length - 1
        tok = self._build_tokenizer_flat(robot_state)
        pro = self._proprio_flat(robot_state, mid)
        if self._split:
            enc_in = self._build_encoder_input(tok)
            token = np.asarray(self._enc_session.run(None, {self._enc_in: enc_in})[0], dtype=np.float32).reshape(-1)
            if self._decoder_in_dim == DEPLOY_DECODER_DIM:
                return self._build_deploy_decoder_input(token, robot_state)
            obs = np.concatenate([token, pro], axis=0).reshape(1, -1)
            return obs
        obs = np.concatenate([tok, pro], axis=0).reshape(1, -1)
        if self.expected_obs_dim > 0 and obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(
                f"SONIC fused ONNX expected obs dim {self.expected_obs_dim}, got {obs.shape[-1]}. "
                f"tokenizer={tok.size}, proprio={pro.size}"
            )
        return obs

    def get_action(self, obs, start_ticker: bool = False):
        if self._split:
            feed = {self._dec_in: obs.astype(np.float32)}
            out = self._dec_session.run(None, feed)[0]
        else:
            feed = {self._in_name: obs.astype(np.float32)}
            out = self.session.run(None, feed)[0]
        action = np.asarray(out, dtype=np.float32).reshape(-1)
        if start_ticker:
            self.ticker += 1
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return action


def _resolve_onnx_paths(cfg: dict) -> tuple[str | None, str | None, str | None]:
    """
    Returns (fused_path, encoder_path, decoder_path).

    Priority: explicit encoder+decoder if both exist; else ``exported_policies/gear_sonic`` defaults;
    else fused ``onnx_model_path`` if that file exists.
    """
    fused = (cfg.get("onnx_model_path") or "").strip()
    enc = (cfg.get("gear_sonic_encoder_onnx_path") or "").strip()
    dec = (cfg.get("gear_sonic_decoder_onnx_path") or "").strip()
    if enc and dec and os.path.isfile(enc) and os.path.isfile(dec):
        return None, enc, dec
    rel = (cfg.get("gear_sonic_deploy_policy_dir") or "").strip()
    if not rel:
        rel = str(_default_gear_sonic_release_dir())
    rel_p = Path(rel).expanduser()
    enc_n = cfg.get("gear_sonic_encoder_onnx_name", "encoder.onnx")
    dec_n = cfg.get("gear_sonic_decoder_onnx_name", "decoder.onnx")
    enc_p = rel_p / enc_n
    dec_p = rel_p / dec_n
    if enc_p.is_file() and dec_p.is_file():
        return None, str(enc_p), str(dec_p)
    if fused and os.path.isfile(fused) and not cfg.get("gear_sonic_force_split", False):
        return fused, None, None
    return None, None, None


def build_gear_sonic_g1_onnx_policy(cfg: dict) -> RLGearSonicG1OnnxPolicy:
    """Construct policy from the ``rl_policy`` subsection of an experiment YAML."""
    djp = cfg.get("gear_sonic_default_joint_pos")
    if djp is not None:
        default_joint_pos = np.array([float(x) for x in djp], dtype=np.float64)
    else:
        default_joint_pos = None
    asp = cfg.get("gear_sonic_action_scale")
    if asp is not None:
        action_scale = np.array([float(x) for x in asp], dtype=np.float64)
    else:
        action_scale = None
    order = cfg.get("gear_sonic_tokenizer_feature_order")
    fused, enc, dec = _resolve_onnx_paths(cfg)
    if fused is None and (enc is None or dec is None):
        rel = cfg.get("gear_sonic_deploy_policy_dir") or str(_default_gear_sonic_release_dir())
        enc_n = cfg.get("gear_sonic_encoder_onnx_name", "encoder.onnx")
        dec_n = cfg.get("gear_sonic_decoder_onnx_name", "decoder.onnx")
        tried_enc = str(Path(rel).expanduser() / enc_n)
        tried_dec = str(Path(rel).expanduser() / dec_n)
        fused_try = (cfg.get("onnx_model_path") or "").strip()
        msg = (
            "No SONIC ONNX found.\n"
            "  • Split (same as gear_sonic_deploy): place encoder + decoder ONNX next to each other, e.g.\n"
            f"      {tried_enc}\n"
            f"      {tried_dec}\n"
            "    (copy from eval ``exported/model_step_*_encoder.onnx`` and ``*_decoder.onnx``), or set\n"
            "    gear_sonic_encoder_onnx_path / gear_sonic_decoder_onnx_path.\n"
            "  • Fused: set onnx_model_path to ``*_g1.onnx`` from ``export_universal_token_module_as_onnx``.\n"
        )
        if fused_try:
            msg += f"  • onnx_model_path was set to {fused_try!r} but file is missing or unreadable.\n"
        raise FileNotFoundError(msg)
    sl = cfg.get("gear_sonic_encoder_tokenizer_slice")
    enc_slice: tuple[int, int] | None = None
    if sl is not None and len(sl) == 2:
        enc_slice = (int(sl[0]), int(sl[1]))

    return RLGearSonicG1OnnxPolicy(
        cfg["ref_motion_path"],
        fused_onnx_path=fused,
        encoder_onnx_path=enc,
        decoder_onnx_path=dec,
        num_future_frames=int(cfg.get("gear_sonic_num_future_frames", 5)),
        lookahead_frame_skips=int(cfg.get("lookahead_frame_skips", 1)),
        init_at_first_frame=bool(cfg.get("init_at_first_frame", False)),
        ref_motion_start_index=int(cfg.get("ref_motion_start_index", 0)),
        num_dof=int(cfg.get("gear_sonic_num_dof", 29)),
        tokenizer_feature_order=order,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        onnx_providers=cfg.get("gear_sonic_onnx_providers"),
        encoder_index=float(cfg.get("gear_sonic_encoder_index", 0.0)),
        encoder_tokenizer_slice=enc_slice,
        omega_is_world_frame=bool(cfg.get("gear_sonic_omega_is_world_frame", False)),
        apply_heading_alignment=bool(
            cfg["gear_sonic_apply_heading_alignment"]
            if "gear_sonic_apply_heading_alignment" in cfg
            else not bool(cfg.get("use_root_state", False))
        ),
        use_history_buffer=bool(cfg.get("gear_sonic_use_history_buffer", True)),
        use_root_state=bool(cfg.get("use_root_state", False)),
    )
