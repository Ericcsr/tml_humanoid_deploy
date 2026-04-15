"""Load Unitree G1 URDF in PyBullet and draw RGB axes at mid360_link (LiDAR frame).

Run from repo root, same as replay_motion.py (meshes resolve relative to the URDF directory):
  python visualize_mid360_pybullet.py

Uses the same PyBullet patterns as replay_motion.py: GUI mode, loadURDF, link names from
getJointInfo; adds a debug-draw loop for the requested link pose.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pybullet as pb


def _default_urdf() -> Path:
    return Path(__file__).resolve().parent / "assets" / "g1" / "g1_29dof_kin_extended.urdf"


def link_index_for_name(robot_id: int, link_name: str) -> int:
    """PyBullet link index: -1 for base link, else joint index of the child link."""
    bi = pb.getBodyInfo(robot_id)
    if bi[0].decode("utf-8") == link_name:
        return -1
    for j in range(pb.getNumJoints(robot_id)):
        child = pb.getJointInfo(robot_id, j)[12].decode("utf-8")
        if child == link_name:
            return j
    raise ValueError(f"Link {link_name!r} not found on robot body {robot_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a named link on Unitree URDF in PyBullet.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=_default_urdf(),
        help="Path to URDF (default: assets/g1/g1_29dof_kin_extended.urdf next to this script)",
    )
    parser.add_argument("--link", type=str, default="mid360_link", help="Link name to visualize")
    parser.add_argument(
        "--axis-length", type=float, default=0.2, help="Debug axis length in meters"
    )
    parser.add_argument(
        "--chdir-urdf",
        action="store_true",
        help="chdir to URDF parent so relative mesh paths match replay_motion cwd behavior",
    )
    args = parser.parse_args()

    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    prev_cwd: str | None = None
    urdf_arg = str(urdf_path)
    if args.chdir_urdf:
        prev_cwd = os.getcwd()
        os.chdir(urdf_path.parent)
        urdf_arg = urdf_path.name

    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.setAdditionalSearchPath(str(urdf_path.parent))
    pb.setGravity(0, 0, 0)

    robot = pb.loadURDF(urdf_arg, useFixedBase=True)

    link_idx = link_index_for_name(robot, args.link)
    print(f"Link {args.link!r} -> PyBullet link index {link_idx}")

    axis_len = float(args.axis_length)
    line_ids = [-1, -1, -1]
    colors = ([1, 0, 0], [0, 1, 0], [0, 0, 1])

    try:
        pb.setRealTimeSimulation(1)
        while pb.isConnected():
            ls = pb.getLinkState(robot, link_idx, computeForwardKinematics=True)
            pos = np.array(ls[4])
            orn = ls[5]
            rot = np.array(pb.getMatrixFromQuaternion(orn)).reshape(3, 3)

            for a in range(3):
                end = pos + rot[:, a] * axis_len
                line_ids[a] = pb.addUserDebugLine(
                    pos.tolist(),
                    end.tolist(),
                    lineColorRGB=colors[a],
                    lineWidth=2.5,
                    replaceItemUniqueId=line_ids[a],
                )
            time.sleep(1.0 / 30.0)
    finally:
        if prev_cwd is not None:
            os.chdir(prev_cwd)
        pb.disconnect()


if __name__ == "__main__":
    main()
