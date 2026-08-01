#!/usr/bin/env python3
"""
MOCK camera publisher — stands in for the (not-yet-built) camera server.

Publishes JPEG bytes to the camera Redis topics the bridge reads
(latest-wins STRING keys), so the bridge's image path is exercised end-to-end.
Frames are synthetic (a moving gradient + frame counter) unless --image is given.

The real camera server must publish the SAME way: raw JPEG bytes under
``camera:<view>`` keys, matching the bridge's --video_topic mapping.
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2
import redis

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _DEPLOY_ROOT)
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402


def _synthetic(w, h, t, tint):
    img = np.zeros((h, w, 3), np.uint8)
    grad = (np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            + np.uint8((t * 5) % 255))
    img[:] = grad[..., None]
    img[..., tint] = 255 - img[..., tint]
    cv2.putText(img, f"{t}", (w // 2 - 40, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 3)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--topics", nargs="+",
                   default=["camera:pelvis_left_view", "camera:pelvis_right_view"],
                   help="Redis keys to publish JPEG bytes to.")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--image", default=None, help="Optional fixed image to serve instead of synthetic.")
    args = p.parse_args()

    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    fixed = None
    if args.image:
        fixed = cv2.imread(args.image)
        if fixed is None:
            raise FileNotFoundError(args.image)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]
    period = 1.0 / args.hz
    print(f"[mock_camera] publishing {args.topics} @ {args.hz}Hz "
          f"{args.width}x{args.height}", flush=True)
    t = 0
    next_tick = time.perf_counter()
    while True:
        for ti, topic in enumerate(args.topics):
            frame = fixed if fixed is not None else _synthetic(args.width, args.height, t, ti % 3)
            ok, buf = cv2.imencode(".jpg", frame, jpeg_params)
            if ok:
                rdb.set(topic, buf.tobytes())
        t += 1
        next_tick += period
        dt = next_tick - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
