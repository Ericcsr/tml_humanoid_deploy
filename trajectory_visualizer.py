"""
Simple matplotlib visualizer for root trajectories of robot vs reference motion.
Shows separate x, y, z error vs time when --metric is used.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_root_trajectories(robot_traj: np.ndarray, ref_traj: np.ndarray):
    """Plot x, y, z position error (robot - ref) vs frame index.

    Args:
        robot_traj: (N, 3) array of robot root positions
        ref_traj: (N, 3) array of reference root positions
    """
    robot_traj = np.asarray(robot_traj)
    ref_traj = np.asarray(ref_traj)

    err = robot_traj - ref_traj
    frames = np.arange(len(robot_traj))

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    for i, (ax, label) in enumerate(zip(axes, ["x", "y", "z"])):
        ax.plot(frames, err[:, i], "b-", linewidth=1.5, alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{label} error (m)")
        ax.set_title(f"Root {label} error (robot - ref)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Root Position Error: Robot vs Reference (init-relative frame)")
    plt.tight_layout()
    plt.show()
