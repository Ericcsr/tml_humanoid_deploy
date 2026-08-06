#!/usr/bin/env python3
"""Extract a qpos motion trajectory from a ROS2/MCAP teleoperation recording.

This reads a Unitree G1 whole-body teleoperation ``.mcap`` (ROS2 + CDR encoding)
and produces the ``qpos`` motion format consumed by ``qpos_to_motion.py`` --
so the pipeline is::

    episode_XXXX.mcap  --(this script)-->  motion_qpos.npz
                       --(qpos_to_motion.py)-->  motion.npz  (tracking/body format)
                       --(run_controller.py)-->  policy reference motion

What it reads
-------------
Two topics are needed to reconstruct the floating-base robot pose per frame:

  * ``/stamped/lowstate``  (``homies/msg/LowStateStamped`` -> ``unitree_hg/LowState``)
        - ``motor_state[0:29].q``: the 29 body-joint angles. The Unitree G1 29-DoF
          motor order (left leg 6, right leg 6, waist 3, left arm 7, right arm 7)
          is identical to ``qpos_to_motion.MUJOCO_JOINT_ORDER`` (MuJoCo/URDF order),
          so no reordering is applied here.
        - ``imu_state.quaternion``: torso IMU orientation (w, x, y, z), used as the
          root orientation fallback when odometry is absent.
        This topic is the highest-rate signal, so it defines the master timeline.

  * ``/lf/odommodestate`` (``unitree_go/msg/SportModeState``)
        - ``position``: root world position (x, y, z).
        - ``imu_state.quaternion``: root world orientation (w, x, y, z).
        Lower rate; linearly interpolated (pos) / slerped (quat) onto the lowstate
        timeline.

Output (qpos format, ready for qpos_to_motion.py)
-------------------------------------------------
A ``.npz`` with:
  * ``qpos``: (T, 36) float64. Default ``pos_first`` layout:
        [x, y, z, qw, qx, qy, qz, joints[29]]   (quaternion is w-first / wxyz)
  * ``fps``: scalar sampling rate of the output timeline.

Frame timestamps come from each message's MCAP ``log_time`` (a uniform, monotonic
clock shared across topics), which is more robust than mixing per-topic header
stamps.

CLI usage
---------
    # extract qpos
    python mcap_to_qpos.py --input episode_0001.mcap --output episode_0001_qpos.npz
    # then convert to tracking/body format:
    python qpos_to_motion.py --input episode_0001_qpos.npz --output episode_0001_motion.npz \\
        --robot-xml assets/g1/scene_29dof.xml --output-fps 50

    # visualize the head camera stream as a video:
    python mcap_to_qpos.py --input episode_0001.mcap --head-camera-video head.mp4

    # visualize the extracted qpos as a MuJoCo rollout video:
    python mcap_to_qpos.py --input episode_0001.mcap --qpos-video qpos.mp4 \\
        --robot-xml assets/g1/scene_29dof.xml
    # (offscreen render needs a GL backend: MUJOCO_GL=glfw with an X display, or egl headless)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Topics we consume. Everything else in the bag (cameras, hands, cmds) is skipped.
LOWSTATE_TOPIC = "/stamped/lowstate"
ODOM_TOPIC = "/lf/odommodestate"

_NUM_JOINTS = 29  # G1 body DoF (first 29 of the 35 motor_state entries)


def _log_time_s(log_time_ns: int) -> float:
    """MCAP log_time is int nanoseconds; return float seconds."""
    return log_time_ns * 1e-9


def read_streams(input_file: str | Path, *, max_frames: int | None = None):
    """Read the lowstate and odom streams from an MCAP file.

    Returns:
        ls_t     (Nl,)    lowstate times [s]
        ls_q     (Nl, 29) joint angles (MuJoCo/URDF order)
        ls_quat  (Nl, 4)  torso IMU quaternion (wxyz)
        od_t     (No,)    odom times [s]
        od_pos   (No, 3)  root world position
        od_quat  (No, 4)  root world quaternion (wxyz)
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    ls_t: list[float] = []
    ls_q: list[list[float]] = []
    ls_quat: list[list[float]] = []
    od_t: list[float] = []
    od_pos: list[list[float]] = []
    od_quat: list[list[float]] = []

    with open(input_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=[LOWSTATE_TOPIC, ODOM_TOPIC]
        ):
            t = _log_time_s(message.log_time)
            if channel.topic == LOWSTATE_TOPIC:
                motors = decoded.data.motor_state
                ls_q.append([float(motors[i].q) for i in range(_NUM_JOINTS)])
                q = decoded.data.imu_state.quaternion  # wxyz
                ls_quat.append([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
                ls_t.append(t)
                if max_frames is not None and len(ls_t) >= max_frames:
                    break
            elif channel.topic == ODOM_TOPIC:
                p = decoded.position
                q = decoded.imu_state.quaternion  # wxyz
                od_pos.append([float(p[0]), float(p[1]), float(p[2])])
                od_quat.append([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
                od_t.append(t)

    if not ls_t:
        raise ValueError(f"No '{LOWSTATE_TOPIC}' messages found in {input_file}")

    return (
        np.asarray(ls_t, dtype=np.float64),
        np.asarray(ls_q, dtype=np.float64),
        np.asarray(ls_quat, dtype=np.float64),
        np.asarray(od_t, dtype=np.float64),
        np.asarray(od_pos, dtype=np.float64),
        np.asarray(od_quat, dtype=np.float64),
    )


def _interp_pos(src_t: np.ndarray, src_pos: np.ndarray, tgt_t: np.ndarray) -> np.ndarray:
    """Per-axis linear interpolation of positions onto tgt_t (clamped at the ends)."""
    return np.stack(
        [np.interp(tgt_t, src_t, src_pos[:, c]) for c in range(src_pos.shape[1])],
        axis=1,
    )


def _interp_quat_wxyz(src_t: np.ndarray, src_quat_wxyz: np.ndarray, tgt_t: np.ndarray) -> np.ndarray:
    """Slerp a quaternion trajectory (wxyz) onto tgt_t (clamped at the ends)."""
    from scipy.spatial.transform import Rotation, Slerp

    if src_quat_wxyz.shape[0] == 1:
        return np.repeat(src_quat_wxyz, tgt_t.shape[0], axis=0)
    rot = Rotation.from_quat(src_quat_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw
    slerp = Slerp(src_t, rot)
    out = slerp(np.clip(tgt_t, src_t[0], src_t[-1])).as_quat()  # xyzw
    return out[:, [3, 0, 1, 2]]  # -> wxyz


def build_qpos(
    ls_t: np.ndarray,
    ls_q: np.ndarray,
    ls_quat: np.ndarray,
    od_t: np.ndarray,
    od_pos: np.ndarray,
    od_quat: np.ndarray,
    *,
    use_odom_orientation: bool = True,
) -> tuple[np.ndarray, float]:
    """Assemble (T, 36) qpos on the lowstate timeline. Returns (qpos, fps)."""
    n = ls_t.shape[0]

    if od_t.shape[0] > 0:
        root_pos = _interp_pos(od_t, od_pos, ls_t)
    else:
        # No odometry in this bag: leave the root at the origin (xy/z unknown).
        root_pos = np.zeros((n, 3), dtype=np.float64)

    if use_odom_orientation and od_t.shape[0] > 0:
        root_quat = _interp_quat_wxyz(od_t, od_quat, ls_t)
    else:
        # Fall back to the torso IMU orientation (already on the lowstate timeline).
        root_quat = ls_quat.copy()

    qpos = np.concatenate([root_pos, root_quat, ls_q], axis=1)  # pos_first, wxyz

    # fps from the median sample spacing of the master timeline.
    if n >= 2:
        dt = float(np.median(np.diff(ls_t)))
        fps = 1.0 / dt if dt > 0 else 0.0
    else:
        fps = 0.0
    return qpos, fps


def convert_mcap_to_qpos(
    input_file: str | Path,
    *,
    use_odom_orientation: bool = True,
    max_frames: int | None = None,
    downsample: int = 1,
) -> dict:
    """Read an MCAP recording and return a qpos motion dict {'qpos', 'fps'}.

    ``downsample`` keeps every Nth lowstate frame (stride) before building qpos;
    the reported ``fps`` is recomputed from the downsampled timeline so it scales
    down accordingly (e.g. downsample=2 halves the effective fps).
    """
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")
    ls_t, ls_q, ls_quat, od_t, od_pos, od_quat = read_streams(
        input_file, max_frames=max_frames
    )
    if downsample > 1:
        ls_t = ls_t[::downsample]
        ls_q = ls_q[::downsample]
        ls_quat = ls_quat[::downsample]
    qpos, fps = build_qpos(
        ls_t, ls_q, ls_quat, od_t, od_pos, od_quat,
        use_odom_orientation=use_odom_orientation,
    )
    return {"qpos": qpos, "fps": np.array([fps], dtype=np.float64)}


# --------------------------------------------------------------------------- #
# Visualization: head camera -> video
# --------------------------------------------------------------------------- #
HEAD_CAMERA_TOPIC = "/camera/head/image/compressed"


def visualize_head_camera(
    input_file: str | Path,
    output_video: str | Path,
    *,
    topic: str = HEAD_CAMERA_TOPIC,
    fps: float | None = None,
    max_frames: int | None = None,
) -> None:
    """Decode a CompressedImage topic and write it out as an .mp4 video.

    ``fps`` defaults to the recording's own frame rate for the topic (derived
    from the median inter-frame log_time spacing).
    """
    import cv2
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    times: list[float] = []
    with open(input_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, _channel, message, decoded in reader.iter_decoded_messages(topics=[topic]):
            buf = np.frombuffer(bytes(decoded.data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                continue
            frames.append(img)
            times.append(_log_time_s(message.log_time))
            if max_frames is not None and len(frames) >= max_frames:
                break

    if not frames:
        raise ValueError(f"No decodable images found on topic '{topic}' in {input_file}")

    if fps is None:
        if len(times) >= 2:
            dt = float(np.median(np.diff(times)))
            fps = 1.0 / dt if dt > 0 else 30.0
        else:
            fps = 30.0

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    try:
        for img in frames:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
    finally:
        writer.release()
    print(
        f"Saved {output_video}  frames={len(frames)}  size={w}x{h}  fps={fps:.2f}  "
        f"duration={len(frames) / max(fps, 1e-9):.1f}s"
    )


# --------------------------------------------------------------------------- #
# Visualization: qpos -> MuJoCo rollout video
# --------------------------------------------------------------------------- #
def visualize_qpos_mujoco(
    qpos: np.ndarray,
    output_video: str | Path,
    robot_xml: str | Path,
    *,
    fps: float = 50.0,
    rot_first: bool = False,
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
) -> None:
    """Render a (T, 36) qpos trajectory to an .mp4.

    Thin wrapper around ``qpos_to_video.render_qpos_video`` so the MuJoCo replay
    logic lives in one place. See that module for backend/GL notes.
    """
    from qpos_to_video import render_qpos_video

    render_qpos_video(
        qpos,
        output_video,
        robot_xml,
        fps=fps,
        rot_first=rot_first,
        width=width,
        height=height,
        camera=camera,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract a (T, 36) qpos motion from a ROS2/MCAP G1 teleop recording."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .mcap recording")
    parser.add_argument("--output", type=Path, default=None, help="Output qpos .npz")
    parser.add_argument(
        "--imu-orientation",
        action="store_true",
        help="Use the torso IMU quaternion for the root orientation instead of odometry.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many lowstate frames (for quick tests).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every Nth frame (stride). e.g. 2 = every other frame; "
        "output fps is scaled down accordingly.",
    )
    parser.add_argument(
        "--head-camera-video",
        type=Path,
        default=None,
        help="Also decode the head camera stream and write it to this .mp4.",
    )
    parser.add_argument(
        "--qpos-video",
        type=Path,
        default=None,
        help="Also render the extracted qpos through MuJoCo and write it to this .mp4.",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("assets/g1/scene_29dof.xml"),
        help="MJCF for --qpos-video rendering (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--video-camera",
        type=str,
        default=None,
        help="Named MuJoCo camera for --qpos-video (default: free/tracking camera).",
    )
    args = parser.parse_args()

    if args.output is None and args.head_camera_video is None and args.qpos_video is None:
        parser.error("nothing to do: pass --output and/or --head-camera-video and/or --qpos-video")

    # (1) Head camera video — independent of qpos extraction.
    if args.head_camera_video is not None:
        visualize_head_camera(
            args.input,
            args.head_camera_video,
            max_frames=args.max_frames,
        )

    # qpos extraction (needed for --output and/or --qpos-video).
    motion = None
    if args.output is not None or args.qpos_video is not None:
        motion = convert_mcap_to_qpos(
            args.input,
            use_odom_orientation=not args.imu_orientation,
            max_frames=args.max_frames,
            downsample=args.downsample,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(args.output), **motion)
        qpos = motion["qpos"]
        print(
            f"Saved {args.output}  qpos={qpos.shape}  fps={float(motion['fps'][0]):g}  "
            f"duration={qpos.shape[0] / max(float(motion['fps'][0]), 1e-9):.1f}s"
        )

    # (2) qpos MuJoCo rollout video.
    if args.qpos_video is not None:
        visualize_qpos_mujoco(
            motion["qpos"],
            args.qpos_video,
            args.robot_xml,
            fps=float(motion["fps"][0]),
            camera=args.video_camera,
        )


if __name__ == "__main__":
    main()
