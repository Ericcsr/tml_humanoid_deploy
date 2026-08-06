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


def plot_eef_errors(
    robot_l: np.ndarray,
    ref_l: np.ndarray,
    robot_r: np.ndarray,
    ref_r: np.ndarray,
    output_png: str = "eef_error.png",
):
    """Plot left/right end-effector (hand) position error (robot - ref) and save a PNG.

    Args:
        robot_l / ref_l: (N, 3) robot / reference LEFT hand positions.
        robot_r / ref_r: (N, 3) robot / reference RIGHT hand positions.
        output_png:      path to save the figure to.
    """
    robot_l, ref_l = np.asarray(robot_l), np.asarray(ref_l)
    robot_r, ref_r = np.asarray(robot_r), np.asarray(ref_r)

    frames = np.arange(len(robot_l))
    err_l = np.linalg.norm(robot_l - ref_l, axis=1)
    err_r = np.linalg.norm(robot_r - ref_r, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # Row 0: left/right position-error magnitude vs frame.
    axes[0].plot(frames, err_l, "b-", linewidth=1.5, alpha=0.85, label="left hand")
    axes[0].plot(frames, err_r, "r-", linewidth=1.5, alpha=0.85, label="right hand")
    axes[0].set_ylabel("position error (m)")
    axes[0].set_title("End-effector position error magnitude (robot - ref)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Rows 1-2: per-axis error for each hand.
    for ax, (robot, ref, name, color) in zip(
        axes[1:], [(robot_l, ref_l, "left", "b"), (robot_r, ref_r, "right", "r")]
    ):
        e = robot - ref
        for i, axis in enumerate(["x", "y", "z"]):
            ax.plot(frames, e[:, i], linewidth=1.2, alpha=0.8, label=f"{axis}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{name} hand error (m)")
        ax.set_title(f"{name.capitalize()} hand per-axis error (robot - ref)")
        ax.legend(loc="upper right", ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(
        f"End-effector tracking error (mean L={err_l.mean():.4f} m, R={err_r.mean():.4f} m)"
    )
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"[eef_error] saved plot -> {output_png}")
    plt.show()
