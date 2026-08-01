#!/usr/bin/env python3
"""
Fixed-frame camera stand-in for the sim dry run.

Extracts ONE frame (default the 20th, index 20) from a training dataset's LEFT and
RIGHT pelvis-view videos and publishes those two fixed JPEGs forever on the camera
Redis topics the bridge reads. This lets the sim dry run exercise the full
obs->GR00T->chunk path with realistic (in-distribution) images while the robot's
motion is entirely driven by the tracking controller — the image is intentionally
constant so any motion comes from the policy, not the pixels.

The frames are re-encoded to JPEG exactly as the real camera server must
(raw JPEG bytes under camera:<view> keys).

Default dataset: gr00t_apartment_stairs_noforce_gma (pelvis stereo, matches the
vlk_apartment_noforce_nofk_lowres_gma_p checkpoint's training distribution).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import cv2
import redis

sys.path.insert(0, __file__.rsplit("/vla_bridge/", 1)[0])
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402

DEFAULT_DATASET = os.path.expanduser(
    "~/Projects/Isaac-GR00T/demo_data/gr00t_apartment_stairs_noforce_gma"
)


def _read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_idx, max(0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed reading frame {idx} of {video_path}")
    return frame, idx  # BGR


def _find_video(dataset, view_key, episode):
    """Locate <dataset>/videos/chunk-XXX/observation.images.<view_key>/episode_*.mp4."""
    pat = os.path.join(dataset, "videos", "chunk-*",
                       f"observation.images.{view_key}", f"episode_{episode:06d}.mp4")
    hits = sorted(glob.glob(pat))
    if not hits:
        # fall back to the first episode available for that view
        pat_any = os.path.join(dataset, "videos", "chunk-*",
                               f"observation.images.{view_key}", "episode_*.mp4")
        hits = sorted(glob.glob(pat_any))
    if not hits:
        raise FileNotFoundError(f"no videos for view {view_key!r} under {dataset}")
    return hits[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="LeRobot dataset root with videos/.../observation.images.<view>/.")
    p.add_argument("--frame_index", type=int, default=20, help="Frame to freeze (default 20).")
    p.add_argument("--episode", type=int, default=0, help="Episode video to pull the frame from.")
    # view_key=camera_topic pairs (GR00T video key -> Redis topic the bridge reads).
    p.add_argument("--left", default="pelvis_left_view=camera:pelvis_left_view")
    p.add_argument("--right", default="pelvis_right_view=camera:pelvis_right_view")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--jpeg_quality", type=int, default=90)
    args = p.parse_args()

    pairs = []
    for spec in (args.left, args.right):
        vkey, topic = spec.split("=", 1)
        pairs.append((vkey, topic))

    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]

    frozen = []  # (topic, jpeg_bytes)
    for vkey, topic in pairs:
        vpath = _find_video(args.dataset, vkey, args.episode)
        frame, used_idx = _read_frame(vpath, args.frame_index)
        ok, buf = cv2.imencode(".jpg", frame, jpeg_params)
        if not ok:
            raise RuntimeError(f"jpeg encode failed for {vkey}")
        frozen.append((topic, buf.tobytes()))
        print(f"[train_frame_cam] {vkey}: frame {used_idx} of "
              f"{os.path.relpath(vpath, args.dataset)}  ({frame.shape[1]}x{frame.shape[0]}) "
              f"-> {topic}", flush=True)

    period = 1.0 / args.hz
    next_tick = time.perf_counter()
    print(f"[train_frame_cam] serving {len(frozen)} fixed frame(s) @ {args.hz:.0f}Hz. "
          "Ctrl-C to stop.", flush=True)
    try:
        while True:
            for topic, jpg in frozen:
                rdb.set(topic, jpg)
            next_tick += period
            dt = next_tick - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("[train_frame_cam] stopped.", flush=True)


if __name__ == "__main__":
    main()
