#!/usr/bin/env python3
"""Render a stored qpos motion (.npz/.npy) to a MuJoCo rollout video.

Standalone companion to ``mcap_to_qpos.py`` / ``qpos_to_motion.py``: given a
qpos file (the ``(T, 36)`` format produced by ``mcap_to_qpos.py``), replay it
through MuJoCo forward kinematics and write an ``.mp4``.

Input
-----
A ``.npz`` with a ``qpos`` (T, 36) array (and optional ``fps``), or a bare
``.npy``/``.npz`` holding a ``(T, 36)`` array. Root layout follows
``qpos_to_motion.parse_qpos``:
    * pos_first (default):  [x, y, z, qw, qx, qy, qz, joints[29]]
    * quat_first (--rot-first / --root-layout quat_first): [qw, qx, qy, qz, x, y, z, joints[29]]

Rendering backend
-----------------
Offscreen rendering needs a GL context. This defaults to ``MUJOCO_GL=glfw``
(works with an X display, e.g. DISPLAY=:1). For a truly headless box set
``MUJOCO_GL=egl`` or ``osmesa`` in the environment before running.

CLI usage
---------
    python qpos_to_video.py --input episode_0001_qpos.npz --output episode_0001_qpos.mp4 \\
        --robot-xml assets/g1/scene_29dof.xml
    # override fps / size / camera:
    python qpos_to_video.py --input m.npz --output m.mp4 --fps 50 --width 960 --height 720 --camera track
    # interactive playback in a live MuJoCo viewer (no video written):
    python qpos_to_video.py --input m.npz --interactive
        # SPACE=pause  .=step forward  ,=step backward  R=restart  Esc=quit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

_QPOS_WIDTH = 36
_DEFAULT_FPS = 50.0


def load_qpos_npz(path: str | Path, default_fps: float = _DEFAULT_FPS):
    """Load a qpos file. Returns ``(qpos (T, 36) float64, fps float)``.

    Accepts an .npz with a ``qpos`` key (+ optional ``fps``), or a bare
    ndarray file of shape (T, 36).
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)

    # Bare ndarray (.npy or unnamed).
    if isinstance(data, np.ndarray) and data.dtype != object:
        qpos = np.asarray(data, dtype=np.float64)
        return qpos, float(default_fps)

    # NpzFile / dict-like.
    if hasattr(data, "files"):
        keys = list(data.files)
        if "qpos" in keys:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        elif len(keys) == 1:
            qpos = np.asarray(data[keys[0]], dtype=np.float64)
        else:
            raise ValueError(f"No 'qpos' array in {path} (keys: {keys})")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in keys else float(default_fps)
        return qpos, fps

    raise ValueError(f"Unrecognized qpos file format: {path}")


def render_qpos_video(
    qpos: np.ndarray,
    output_video: str | Path,
    robot_xml: str | Path,
    *,
    fps: float = _DEFAULT_FPS,
    rot_first: bool = False,
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
    frame_range: tuple[int, int] | None = None,
) -> None:
    """Render a (T, 36) qpos trajectory through MuJoCo FK and write an .mp4.

    Reuses ``qpos_to_motion``'s model loading and qpos assembly so the joint /
    root layout exactly matches the extraction pipeline.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import cv2
    import mujoco

    from qpos_to_motion import load_model, parse_qpos, _build_mujoco_qpos

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < _QPOS_WIDTH:
        raise ValueError(f"qpos must be (T, >= {_QPOS_WIDTH}); got {qpos.shape}")
    if frame_range is not None:
        s, e = frame_range
        qpos = qpos[s:e + 1]

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(robot_xml)
    root_pos, root_quat, joint_pos_mujoco = parse_qpos(qpos, rot_first=rot_first)
    mj_qpos = _build_mujoco_qpos(model, root_pos, root_quat, joint_pos_mujoco)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Round fps: the mp4v/MPEG-4 muxer rejects a timebase denominator > 65535,
    # which a raw rate like 100.067 (-> 1000/100067) would produce.
    fps = round(float(fps), 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {output_video} (fps={fps}, size={width}x{height})"
        )
    n = mj_qpos.shape[0]
    try:
        for i in range(n):
            data.qpos[:] = mj_qpos[i]
            mujoco.mj_forward(model, data)
            if camera is not None:
                renderer.update_scene(data, camera=camera)
            else:
                renderer.update_scene(data)
            rgb = renderer.render()  # (H, W, 3) RGB
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    print(
        f"Saved {output_video}  frames={n}  size={width}x{height}  fps={fps:.2f}  "
        f"duration={n / max(fps, 1e-9):.1f}s"
    )


def view_qpos_interactive(
    qpos: np.ndarray,
    robot_xml: str | Path,
    *,
    fps: float = _DEFAULT_FPS,
    rot_first: bool = False,
    camera: str | None = None,
    frame_range: tuple[int, int] | None = None,
    loop: bool = True,
) -> None:
    """Play a (T, 36) qpos trajectory in an interactive MuJoCo viewer window.

    Opens ``mujoco.viewer.launch_passive`` and steps through the trajectory in
    real time (paced by ``fps``). Keyboard controls:
        * SPACE  pause / resume
        * .      step one frame forward (while paused)
        * ,      step one frame backward (while paused)
        * R      restart from the first frame
    Closing the window (or Esc) exits.
    """
    # Interactive viewer needs an on-screen GL context.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import time

    import mujoco
    import mujoco.viewer

    from qpos_to_motion import load_model, parse_qpos, _build_mujoco_qpos

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < _QPOS_WIDTH:
        raise ValueError(f"qpos must be (T, >= {_QPOS_WIDTH}); got {qpos.shape}")
    if frame_range is not None:
        s, e = frame_range
        qpos = qpos[s:e + 1]

    model = load_model(robot_xml)
    root_pos, root_quat, joint_pos_mujoco = parse_qpos(qpos, rot_first=rot_first)
    mj_qpos = _build_mujoco_qpos(model, root_pos, root_quat, joint_pos_mujoco)

    data = mujoco.MjData(model)
    n = mj_qpos.shape[0]
    dt = 1.0 / max(float(fps), 1e-9)

    state = {"i": 0, "paused": False, "step": 0}

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode)
        except ValueError:
            key = ""
        if keycode == 32:  # SPACE
            state["paused"] = not state["paused"]
        elif key in (".", ">"):
            state["step"] = 1
        elif key in (",", "<"):
            state["step"] = -1
        elif key in ("r", "R"):
            state["i"] = 0

    print(
        f"Interactive playback  frames={n}  fps={fps:.2f}  "
        f"[SPACE=pause  .=step+  ,=step-  R=restart  Esc=quit]"
    )
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        if camera is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            try:
                viewer.cam.trackbodyid = model.camera(camera).targetbodyid
            except Exception:
                pass
        while viewer.is_running():
            tic = time.perf_counter()
            i = state["i"] % n
            data.qpos[:] = mj_qpos[i]
            mujoco.mj_forward(model, data)
            viewer.sync()

            if state["paused"]:
                if state["step"]:
                    state["i"] = (i + state["step"]) % n
                    state["step"] = 0
                time.sleep(0.01)
                continue

            nxt = i + 1
            if nxt >= n and not loop:
                state["paused"] = True
                continue
            state["i"] = nxt % n

            elapsed = time.perf_counter() - tic
            if elapsed < dt:
                time.sleep(dt - elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Render a stored qpos (.npz/.npy) motion to a MuJoCo video."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input qpos .npz/.npy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 (required unless --interactive).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a live MuJoCo viewer and play the motion instead of writing a video.",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="With --interactive, stop at the last frame instead of looping.",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("assets/g1/scene_29dof.xml"),
        help="MJCF relative to repo root or absolute.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video fps (default: fps stored in the file, else 50).",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Named MuJoCo camera (default: free/tracking camera).",
    )
    parser.add_argument(
        "--root-layout",
        choices=["pos_first", "quat_first"],
        default=None,
        help="qpos root layout for the first 7 entries "
        "('pos_first' default, 'quat_first' = quaternion-first). Overrides --rot-first.",
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
        help="Inclusive frame range to render.",
    )
    args = parser.parse_args()

    if args.root_layout is not None:
        rot_first = args.root_layout == "quat_first"
    else:
        rot_first = args.rot_first

    qpos, file_fps = load_qpos_npz(args.input)
    fps = float(args.fps) if args.fps is not None else file_fps
    frame_range = tuple(args.frame_range) if args.frame_range else None

    if args.interactive:
        view_qpos_interactive(
            qpos,
            args.robot_xml,
            fps=fps,
            rot_first=rot_first,
            camera=args.camera,
            frame_range=frame_range,
            loop=not args.no_loop,
        )
        return

    if args.output is None:
        parser.error("--output is required unless --interactive is set.")

    render_qpos_video(
        qpos,
        args.output,
        args.robot_xml,
        fps=fps,
        rot_first=rot_first,
        width=args.width,
        height=args.height,
        camera=args.camera,
        frame_range=frame_range,
    )


if __name__ == "__main__":
    main()
