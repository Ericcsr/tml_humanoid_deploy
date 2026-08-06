#!/usr/bin/env python3
"""Read a rendered motion.npz, reorder joints to the robot's order, resample to a target fps, and save.

Input qpos is (T, 40) = 7 base (3 pos + 4 quat) + 33 joints. The input joints are stored in the
order given by the file's ``joint_names`` (a type-grouped order, e.g. all *_hip_pitch first), which
is **not** the order the robot / downstream FK expects.

We reorder the joint columns by name into the MuJoCo/URDF depth-first order used everywhere else
(``qpos_to_motion.MUJOCO_JOINT_ORDER``), which also drops the 4 gripper columns (they aren't in the
29-DoF order). Output is (T, 36) = 7 base + 29 joints in robot order.

Resampling keeps the same wall-clock duration: base position and joint angles are linearly
interpolated; the base orientation quaternion (columns 3:7) is spherically interpolated (slerp).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qpos_to_motion import MUJOCO_JOINT_ORDER


BASE_COLS = 7  # 3 pos + 4 quat
KEEP_COLS = BASE_COLS + len(MUJOCO_JOINT_ORDER)  # 7 base + 29 joints = 36
QUAT_SLICE = slice(3, 7)  # base orientation quaternion columns within qpos


def reorder_joints(qpos: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Reorder the joint columns of a (T, 7 + J) qpos into robot (MuJoCo/URDF) order.

    ``joint_names`` are the names for the joint columns of ``qpos`` (i.e. columns
    ``BASE_COLS : BASE_COLS + len(joint_names)``), in the file's own storage order.
    Returns (T, 36) = base(7) + the 29 ``MUJOCO_JOINT_ORDER`` joints, dropping any
    joints (e.g. grippers) not in that order.
    """
    name_to_col = {n: i for i, n in enumerate(joint_names)}
    missing = [n for n in MUJOCO_JOINT_ORDER if n not in name_to_col]
    if missing:
        raise SystemExit(
            f"Input joint_names is missing {len(missing)} robot joint(s): {missing}"
        )
    src_cols = [BASE_COLS + name_to_col[n] for n in MUJOCO_JOINT_ORDER]
    base = qpos[:, :BASE_COLS]
    joints = qpos[:, src_cols]
    return np.concatenate([base, joints], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        default="vlk_gripper_rendered_data_v1.1/BEAR/0000/motion.npz",
        help="Path to input motion.npz.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Default: <input_dir>/motion_36_<target_fps>fps.npz",
    )
    p.add_argument("--target-fps", type=int, required=True, help="Target fps to resample qpos to.")
    return p.parse_args()


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (order-agnostic 4-vectors)."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    # Take the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Nearly parallel: linear interpolate and renormalize.
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_0 = np.sin(theta_0)
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return s0 * q0 + s1 * q1


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample (T, C) qpos from src_fps to dst_fps, keeping the same duration.

    Columns 3:7 are treated as a base orientation quaternion (slerp); all others are linear.
    """
    t_src = qpos.shape[0]
    if t_src < 2 or src_fps == dst_fps:
        return qpos.copy()

    duration = (t_src - 1) / src_fps
    t_dst = int(round(duration * dst_fps)) + 1
    # Sample times (seconds) for source and destination frames.
    src_times = np.arange(t_src) / src_fps
    dst_times = np.linspace(0.0, duration, t_dst)

    out = np.empty((t_dst, qpos.shape[1]), dtype=qpos.dtype)

    # Linear interpolation for every column (overwritten for the quaternion below).
    for c in range(qpos.shape[1]):
        out[:, c] = np.interp(dst_times, src_times, qpos[:, c])

    # Slerp the quaternion columns if present.
    if qpos.shape[1] >= QUAT_SLICE.stop:
        quats = qpos[:, QUAT_SLICE]
        for i, ti in enumerate(dst_times):
            # Locate the source interval [j, j+1] containing ti.
            j = int(np.searchsorted(src_times, ti, side="right") - 1)
            j = min(max(j, 0), t_src - 2)
            seg = src_times[j + 1] - src_times[j]
            frac = 0.0 if seg <= 0 else (ti - src_times[j]) / seg
            out[i, QUAT_SLICE] = slerp(quats[j], quats[j + 1], float(frac))

    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser()
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    data = np.load(in_path, allow_pickle=True)
    if "qpos" not in data.files:
        raise SystemExit(f"'qpos' not in {in_path} (keys: {list(data.files)})")

    qpos = np.asarray(data["qpos"])
    src_fps = int(data["fps"]) if "fps" in data.files else None
    if src_fps is None:
        raise SystemExit("Input has no 'fps'; cannot resample. Provide the source fps.")

    if "joint_names" not in data.files:
        raise SystemExit(
            f"'joint_names' not in {in_path} (keys: {list(data.files)}); "
            "cannot align joints to robot order."
        )
    joint_names = [str(n) for n in np.asarray(data["joint_names"])]
    if qpos.shape[1] < BASE_COLS + len(joint_names):
        raise SystemExit(
            f"qpos has {qpos.shape[1]} columns, fewer than base(7) + "
            f"{len(joint_names)} joints."
        )

    # (1) Reorder joints from the file's storage order into robot (MuJoCo/URDF) order.
    #     This also drops joints not in MUJOCO_JOINT_ORDER (e.g. the 4 gripper columns).
    qpos = reorder_joints(qpos, joint_names)
    print(f"[proc] input qpos={np.asarray(data['qpos']).shape} src_fps={src_fps} "
          f"joints={len(joint_names)} -> reordered to robot order (T, {KEEP_COLS})")

    # (2) Resample to target fps.
    qpos_out = resample_qpos(qpos, float(src_fps), float(args.target_fps))
    print(f"[proc] resampled {qpos.shape[0]} -> {qpos_out.shape[0]} frames "
          f"({src_fps} -> {args.target_fps} fps, duration "
          f"{(qpos.shape[0]-1)/src_fps:.3f}s preserved)")

    out_path = (
        Path(args.output).expanduser()
        if args.output
        else in_path.parent / f"motion_{KEEP_COLS}_{args.target_fps}fps.npz"
    )
    np.savez(out_path, qpos=qpos_out.astype(np.float32), fps=np.int64(args.target_fps))
    print(f"[proc] saved qpos={qpos_out.shape} fps={args.target_fps} -> {out_path}")


if __name__ == "__main__":
    main()
