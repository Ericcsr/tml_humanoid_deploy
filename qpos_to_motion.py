#!/usr/bin/env python3
"""Convert retargeted MuJoCo-qpos motion to tracking/body motion data (no IsaacLab).

This is a standalone, importable reimplementation of the conversion done by
``bm_generalist/scripts/omni_to_npz.py`` -- but using **MuJoCo forward kinematics**
instead of IsaacLab, so it has no Isaac Sim / isaaclab dependency.

Input (qpos format, same as omni_to_npz input)
----------------------------------------------
A ``.npz``/``.npy`` with:
  - ``qpos``: (T, 36) float. The first 7 entries are the floating-base root; the layout
    is selectable:
        * pos_first (default, ``rot_first=False``):  [x, y, z, qw, qx, qy, qz, joints[29]]
        * quat_first (``rot_first=True``):           [qw, qx, qy, qz, x, y, z, joints[29]]
    Quaternions are w-first (wxyz). Joints (29) are in MuJoCo / URDF (depth-first) order.
  - ``fps``: scalar (optional; falls back to ``control_dt`` or a provided default).

Output (training / tracking format, same schema as omni_to_npz output)
----------------------------------------------------------------------
A dict (also saveable as ``.npz``) with:
  - ``fps``
  - ``joint_pos`` / ``joint_vel``: (T, 29) in **IsaacLab joint order**
  - ``body_pos_w``:  (T, B, 3)  world position,  index 0 = root (pelvis)
  - ``body_quat_w``: (T, B, 4)  world quaternion (w, x, y, z), index 0 = root
  - ``body_lin_vel_w`` / ``body_ang_vel_w``: (T, B, 3) world body velocities
  - ``body_names``: (B,) the body name for each column (self-describing order)

Joint order
-----------
The qpos joints are in MuJoCo / URDF (depth-first) order. IsaacLab uses a
breadth-first (BFS) order. ``joint_pos_isaac = joint_pos_mujoco[:, MUJOCO_TO_ISAAC_DOF]``.

Body order
----------
IsaacLab orders articulation bodies breadth-first from the root. Doing a BFS over
the MuJoCo body tree (children visited in ascending body-id == XML definition order)
reproduces the IsaacLab body order. This is computed from the loaded model in
``get_isaaclab_body_order`` and validated against ``ISAACLAB_G1_BODY_ORDER``.

Velocities are computed by finite differences (lin: gradient; ang: SO3 central
difference), which is self-consistent with the FK body trajectories.

CLI usage (run from the deploy repo root, or pass an absolute --robot-xml)
-------------------------------------------------------------------------
    python qpos_to_motion.py --input motion_qpos.npz --output motion.npz \\
        --robot-xml assets/g1/scene_29dof.xml --output-fps 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# MuJoCo joint order (URDF / depth-first) for the G1 29-DoF robot. This is the order
# the input qpos joints are stored in, and the order of hinge joints in the MJCF.
MUJOCO_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# MUJOCO_TO_ISAAC_DOF[isaac_idx] = mujoco_idx, i.e. joint_isaac = joint_mujoco[:, MUJOCO_TO_ISAAC_DOF].
# Identical to bm_generalist omni_to_npz MUJOCO_TO_ISACLAB_DOF and deploy utils.params.MUJOCO_TO_ISAAC.
MUJOCO_TO_ISAAC_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
# Inverse: ISAAC_TO_MUJOCO_DOF[mujoco_idx] = isaac_idx, i.e. joint_mujoco = joint_isaac[:, ISAAC_TO_MUJOCO_DOF].
ISAAC_TO_MUJOCO_DOF = np.argsort(MUJOCO_TO_ISAAC_DOF)

# IsaacLab G1 body order (BFS over the kinematic tree). Index 0 is the root (pelvis).
# head_link / logo_link are fixed visual geoms (not separate bodies), so there are 30 bodies.
ISAACLAB_G1_BODY_ORDER = (
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)

_QPOS_WIDTH = 36  # 3 (pos) + 4 (quat) + 29 (joints)
_NUM_JOINTS = 29


# --------------------------------------------------------------------------- #
# Format detection / IO
# --------------------------------------------------------------------------- #
def is_qpos_motion(data) -> bool:
    """Return True if ``data`` looks like raw qpos motion (needs conversion).

    Accepts an ``np.load`` result (NpzFile / 0-d object array / ndarray) or a dict.
    qpos format has a ``qpos`` key and lacks the converted ``body_pos_w`` field; a
    bare ``(T, 36)`` ndarray is also treated as qpos.
    """
    if isinstance(data, np.ndarray) and data.dtype != object:
        return data.ndim == 2 and data.shape[1] == _QPOS_WIDTH

    keys = _data_keys(data)
    if keys is None:
        return False
    has_qpos = "qpos" in keys
    has_converted = "body_pos_w" in keys or "joint_pos" in keys
    return has_qpos and not has_converted


def _data_keys(data):
    """Best-effort key list for NpzFile / dict / 0-d object ndarray; None if not keyed."""
    if hasattr(data, "files"):  # NpzFile
        return list(data.files)
    if isinstance(data, dict):
        return list(data.keys())
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        inner = data.item()
        if isinstance(inner, dict):
            return list(inner.keys())
    if hasattr(data, "keys"):
        try:
            return list(data.keys())
        except Exception:
            return None
    return None


def _data_get(data, key, default=None):
    if hasattr(data, "files"):
        return data[key] if key in data.files else default
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        inner = data.item()
        return inner.get(key, default) if isinstance(inner, dict) else default
    if isinstance(data, dict):
        return data.get(key, default)
    try:
        return data[key]
    except Exception:
        return default


def load_qpos_motion(path: str | Path, default_fps: float = 30.0):
    """Load a qpos motion file. Returns ``(qpos (T, 36) float64, fps float)``."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype != object:
        return np.asarray(data, dtype=np.float64), float(default_fps)

    qpos = _data_get(data, "qpos")
    if qpos is None:
        keys = _data_keys(data)
        raise ValueError(f"No 'qpos' in {path} (keys: {keys})")
    qpos = np.asarray(qpos, dtype=np.float64)

    fps = _data_get(data, "fps")
    if fps is None:
        control_dt = _data_get(data, "control_dt")
        fps = (1.0 / float(np.asarray(control_dt).reshape(-1)[0])) if control_dt is not None else default_fps
    else:
        fps = float(np.asarray(fps).reshape(-1)[0])
    return qpos, float(fps)


# --------------------------------------------------------------------------- #
# qpos parsing / joint reordering
# --------------------------------------------------------------------------- #
def parse_qpos(qpos: np.ndarray, rot_first: bool = False):
    """Split (T, 36) qpos into root_pos (T,3), root_quat_wxyz (T,4), joint_pos_mujoco (T,29)."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < _QPOS_WIDTH:
        raise ValueError(f"qpos must be (T, >= {_QPOS_WIDTH}); got {qpos.shape}")
    if rot_first:
        root_quat = qpos[:, 0:4]
        root_pos = qpos[:, 4:7]
    else:
        root_pos = qpos[:, 0:3]
        root_quat = qpos[:, 3:7]
    joint_pos_mujoco = qpos[:, 7:7 + _NUM_JOINTS]
    return root_pos.copy(), root_quat.copy(), joint_pos_mujoco.copy()


def mujoco_joints_to_isaac(joint_pos_mujoco: np.ndarray) -> np.ndarray:
    """(T, 29) MuJoCo joint order -> (T, 29) IsaacLab joint order."""
    joint_pos_mujoco = np.asarray(joint_pos_mujoco)
    if joint_pos_mujoco.shape[-1] != _NUM_JOINTS:
        raise ValueError(f"expected last dim {_NUM_JOINTS}, got {joint_pos_mujoco.shape}")
    return joint_pos_mujoco[..., MUJOCO_TO_ISAAC_DOF]


# --------------------------------------------------------------------------- #
# MuJoCo model helpers (lazy import so the module loads without mujoco)
# --------------------------------------------------------------------------- #
def load_model(robot_xml: str | Path, repo_root: str | Path | None = None):
    """Compile an MJCF the same way the simulator does (cwd = repo root for meshdir)."""
    import mujoco

    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    path = Path(robot_xml)
    if not path.is_absolute():
        path = (repo / path)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot XML not found: {path}")
    old_cwd = os.getcwd()
    try:
        os.chdir(repo)
        with open(path, "r") as f:
            xml = f.read()
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(old_cwd)
    return model


def get_isaaclab_body_order(model) -> list[str]:
    """BFS body order (children in ascending body-id) == IsaacLab order. Excludes 'world'."""
    import mujoco

    children: dict[int, list[int]] = {}
    for bid in range(1, model.nbody):  # skip world (0)
        parent = int(model.body_parentid[bid])
        children.setdefault(parent, []).append(bid)
    for plist in children.values():
        plist.sort()  # ascending body id == XML definition order

    # Root = the body whose parent is world (0).
    roots = sorted(children.get(0, []))
    order: list[str] = []
    queue = list(roots)
    while queue:
        bid = queue.pop(0)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        order.append(name)
        queue.extend(children.get(bid, []))
    return order


def _build_mujoco_qpos(model, root_pos, root_quat_wxyz, joint_pos_mujoco) -> np.ndarray:
    """Assemble (T, model.nq) qpos: free-joint root + 29 hinges placed by joint name."""
    import mujoco

    t = root_pos.shape[0]
    out = np.zeros((t, model.nq), dtype=np.float64)

    # Free (floating base) joint: 3 pos + 4 quat (wxyz).
    free_adr = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_adr = int(model.jnt_qposadr[j])
            break
    if free_adr is None:
        raise ValueError("Model has no free joint for the floating base.")
    out[:, free_adr:free_adr + 3] = root_pos
    out[:, free_adr + 3:free_adr + 7] = root_quat_wxyz

    # Hinge joints addressed by name (robust to model layout).
    name_to_adr = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jname is not None:
                name_to_adr[jname] = int(model.jnt_qposadr[j])

    for k, jname in enumerate(MUJOCO_JOINT_ORDER):
        adr = name_to_adr.get(jname)
        if adr is None:
            raise ValueError(f"Joint '{jname}' not found in model.")
        out[:, adr] = joint_pos_mujoco[:, k]
    return out


def forward_kinematics(model, mj_qpos: np.ndarray, body_order: list[str]):
    """Per-frame FK. Returns (body_pos_w (T,B,3), body_quat_w (T,B,4) wxyz)."""
    import mujoco

    data = mujoco.MjData(model)
    t = mj_qpos.shape[0]
    b = len(body_order)
    body_ids = []
    for name in body_order:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    body_pos_w = np.zeros((t, b, 3), dtype=np.float64)
    body_quat_w = np.zeros((t, b, 4), dtype=np.float64)
    for i in range(t):
        data.qpos[:] = mj_qpos[i]
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(body_ids):
            body_pos_w[i, k] = data.xpos[bid]
            body_quat_w[i, k] = data.xquat[bid]  # wxyz
    return body_pos_w, body_quat_w


# --------------------------------------------------------------------------- #
# Interpolation / velocities
# --------------------------------------------------------------------------- #
def _resample(root_pos, root_quat_wxyz, joint_pos_mujoco, input_fps, output_fps):
    """Resample to output_fps with lerp (pos/joints) + slerp (root quat). Matches omni_to_npz."""
    from scipy.spatial.transform import Rotation, Slerp

    n = root_pos.shape[0]
    if n < 2 or output_fps is None or abs(output_fps - input_fps) < 1e-9:
        return root_pos, root_quat_wxyz, joint_pos_mujoco

    duration = (n - 1) / input_fps
    out_dt = 1.0 / output_fps
    times = np.arange(0.0, duration, out_dt)
    src_times = np.arange(n) / input_fps

    new_pos = np.stack([np.interp(times, src_times, root_pos[:, c]) for c in range(3)], axis=1)
    new_joints = np.stack(
        [np.interp(times, src_times, joint_pos_mujoco[:, c]) for c in range(joint_pos_mujoco.shape[1])],
        axis=1,
    )
    # scipy quat is xyzw; stored is wxyz.
    rot = Rotation.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]])
    new_rot = Slerp(src_times, rot)(np.clip(times, src_times[0], src_times[-1]))
    q_xyzw = new_rot.as_quat()
    new_quat = q_xyzw[:, [3, 0, 1, 2]]
    return new_pos, new_quat, new_joints


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def _so3_derivative(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion trajectory (central difference)."""
    from scipy.spatial.transform import Rotation

    n = quat_wxyz.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float64)
    r = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])  # to xyzw
    r_prev = r[:-2]
    r_next = r[2:]
    omega = (r_next * r_prev.inv()).as_rotvec() / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def _body_angular_velocities(body_quat_w: np.ndarray, dt: float) -> np.ndarray:
    t, b = body_quat_w.shape[0], body_quat_w.shape[1]
    out = np.zeros((t, b, 3), dtype=np.float64)
    for k in range(b):
        out[:, k, :] = _so3_derivative(body_quat_w[:, k, :], dt)
    return out


# --------------------------------------------------------------------------- #
# Main conversion entry points
# --------------------------------------------------------------------------- #
def convert_qpos_to_motion(
    qpos: np.ndarray,
    fps: float,
    model,
    *,
    rot_first: bool = False,
    output_fps: float | None = None,
    body_order: list[str] | None = None,
    compute_velocities: bool = True,
) -> dict:
    """Convert (T, 36) qpos to the tracking/body motion dict using MuJoCo FK.

    Args:
        qpos: (T, 36) motion. See module docstring for layout.
        fps: input sampling rate (Hz).
        model: a compiled ``mujoco.MjModel`` (see ``load_model``).
        rot_first: True if qpos root is quaternion-first.
        output_fps: if given and != fps, resample before FK (lerp/slerp).
        body_order: explicit body-name order; default = BFS (IsaacLab) order from model.
        compute_velocities: compute joint/body velocities via finite differences.

    Returns:
        dict with fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
        body_lin_vel_w, body_ang_vel_w, body_names.
    """
    root_pos, root_quat, joint_pos_mujoco = parse_qpos(qpos, rot_first=rot_first)

    out_fps = float(fps if output_fps is None else output_fps)
    root_pos, root_quat, joint_pos_mujoco = _resample(
        root_pos, root_quat, joint_pos_mujoco, float(fps), out_fps
    )
    dt = 1.0 / out_fps

    if body_order is None:
        body_order = get_isaaclab_body_order(model)
        if list(body_order) != list(ISAACLAB_G1_BODY_ORDER) and len(body_order) == len(ISAACLAB_G1_BODY_ORDER):
            print(
                "[qpos_to_motion] Warning: BFS body order from model differs from the "
                "reference ISAACLAB_G1_BODY_ORDER; using the model-derived order."
            )

    joint_pos_isaac = mujoco_joints_to_isaac(joint_pos_mujoco)
    mj_qpos = _build_mujoco_qpos(model, root_pos, root_quat, joint_pos_mujoco)
    body_pos_w, body_quat_w = forward_kinematics(model, mj_qpos, body_order)

    result = {
        "fps": np.array([out_fps], dtype=np.float64),
        "joint_pos": joint_pos_isaac.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_names": np.array(list(body_order)),
    }
    if compute_velocities:
        result["joint_vel"] = _gradient(joint_pos_isaac, dt).astype(np.float32)
        result["body_lin_vel_w"] = _gradient(body_pos_w, dt).astype(np.float32)
        result["body_ang_vel_w"] = _body_angular_velocities(body_quat_w, dt).astype(np.float32)
    else:
        z1 = np.zeros_like(joint_pos_isaac, dtype=np.float32)
        z3 = np.zeros_like(body_pos_w, dtype=np.float32)
        result["joint_vel"] = z1
        result["body_lin_vel_w"] = z3
        result["body_ang_vel_w"] = z3.copy()
    return result


def convert_qpos_file_to_motion(
    input_file: str | Path,
    robot_xml: str | Path,
    *,
    rot_first: bool = False,
    output_fps: float | None = None,
    default_fps: float = 30.0,
    frame_range: tuple[int, int] | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Load a qpos file, compile the model, and return the motion dict."""
    qpos, fps = load_qpos_motion(input_file, default_fps=default_fps)
    if frame_range is not None:
        s, e = frame_range
        qpos = qpos[s:e + 1]
    model = load_model(robot_xml, repo_root=repo_root)
    return convert_qpos_to_motion(
        qpos, fps, model, rot_first=rot_first, output_fps=output_fps
    )


def save_motion_npz(path: str | Path, motion: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **motion)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MuJoCo-qpos motion to tracking/body motion data (MuJoCo FK, no IsaacLab)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input qpos .npz/.npy")
    parser.add_argument("--output", type=Path, required=True, help="Output motion .npz")
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("assets/g1/scene_29dof.xml"),
        help="MJCF relative to repo root or absolute",
    )
    parser.add_argument("--output-fps", type=float, default=None, help="Resample to this fps (default: keep input)")
    parser.add_argument("--default-fps", type=float, default=30.0, help="Fallback fps if missing in file")
    parser.add_argument(
        "--root-layout",
        choices=["pos_first", "quat_first"],
        default=None,
        help="qpos root layout for the first 7 entries: "
        "'pos_first' = [x, y, z, qw, qx, qy, qz] (default), "
        "'quat_first' = [qw, qx, qy, qz, x, y, z]. Overrides --rot-first if both are given.",
    )
    parser.add_argument(
        "--rot-first",
        action="store_true",
        help="Alias for --root-layout quat_first (qpos root is quaternion-first).",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Inclusive frame range applied to the input qpos",
    )
    args = parser.parse_args()

    if args.root_layout is not None:
        rot_first = args.root_layout == "quat_first"
    else:
        rot_first = args.rot_first

    motion = convert_qpos_file_to_motion(
        args.input,
        args.robot_xml,
        rot_first=rot_first,
        output_fps=args.output_fps,
        default_fps=args.default_fps,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
    )
    save_motion_npz(args.output, motion)
    print(
        f"Saved {args.output}  joint_pos={motion['joint_pos'].shape}  "
        f"body_pos_w={motion['body_pos_w'].shape}  fps={float(motion['fps'][0]):g}  "
        f"bodies={len(motion['body_names'])}"
    )


if __name__ == "__main__":
    main()
