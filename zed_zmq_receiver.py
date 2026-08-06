#!/usr/bin/env python3
"""Receive synchronized ZED stereo frames (left + right RGB) over ZeroMQ.

Message layout produced by the stereo sender:
    [frame_id (u64)][timestamp_ns (u64)][left RGB bytes][right RGB bytes]
Both frames are HEIGHT x WIDTH x 3 uint8 (RGB). Defaults match the sender's HD1080 (1080x1920).
"""

import argparse
import struct
import time

import numpy as np
import zmq


CHANNELS = 3
HEADER = struct.Struct("!QQ")  # frame_id, capture timestamp_ns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--connect",
        required=True,
        help="Sender address, for example tcp://10.180.152.125:5555",
    )
    # Sender uses HD1080 (1080x1920). Override if the sender's resolution changes.
    parser.add_argument("--height", type=int, default=1080, help="Per-eye frame height (default HD1080: 1080).")
    parser.add_argument("--width", type=int, default=1920, help="Per-eye frame width (default HD1080: 1920).")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Real-time display: left|right merged side-by-side in one OpenCV window "
        "(needs opencv + a display). Press 'q' or Esc to quit.",
    )
    parser.add_argument(
        "--viz-width",
        type=int,
        default=1600,
        help="Total width (px) of the merged --viz window; frames are scaled to fit (default 1600).",
    )
    return parser.parse_args()


def receive_stereo(socket, height, width, frame_bytes, message_bytes):
    """Blocking receive of one stereo message -> (frame_id, timestamp_ns, left, right)."""
    message = socket.recv()
    return _unpack_stereo(message, height, width, frame_bytes, message_bytes)


def try_receive_stereo(socket, height, width, frame_bytes, message_bytes):
    """Non-blocking-ish receive: returns None if no message arrived before RCVTIMEO."""
    try:
        message = socket.recv()
    except zmq.Again:
        return None
    return _unpack_stereo(message, height, width, frame_bytes, message_bytes)


def _unpack_stereo(message, height, width, frame_bytes, message_bytes):
    if len(message) != message_bytes:
        raise ValueError(
            f"Unexpected message size: {len(message)} != {message_bytes} "
            f"(expected two {height}x{width}x{CHANNELS} frames + {HEADER.size}B header). "
            f"Check --height/--width match the sender's resolution."
        )
    frame_id, timestamp_ns = HEADER.unpack_from(message)
    flat = np.frombuffer(message, dtype=np.uint8, offset=HEADER.size)
    left = flat[:frame_bytes].reshape(height, width, CHANNELS)
    right = flat[frame_bytes:].reshape(height, width, CHANNELS)
    return frame_id, timestamp_ns, left, right


def process_frames(left: np.ndarray, right: np.ndarray, frame_id: int, timestamp_ns: int) -> None:
    """Replace this function body with model inference. Receives the left and right RGB frames."""
    pass


def merge_side_by_side(cv2, left, right, total_width, fps, frame_id):
    """Merge left|right (RGB) into one BGR image scaled to total_width, with an overlay label."""
    # Sender frames are RGB; OpenCV displays BGR.
    lb = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    rb = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    h, w = lb.shape[:2]
    per_eye_w = max(1, total_width // 2)
    scale = per_eye_w / w
    new_size = (per_eye_w, max(1, int(round(h * scale))))
    lb = cv2.resize(lb, new_size, interpolation=cv2.INTER_AREA)
    rb = cv2.resize(rb, new_size, interpolation=cv2.INTER_AREA)
    canvas = cv2.hconcat([lb, rb])
    # Divider + labels.
    cv2.line(canvas, (per_eye_w, 0), (per_eye_w, canvas.shape[0]), (60, 60, 60), 1)
    cv2.putText(canvas, "LEFT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT", (per_eye_w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, f"id={frame_id} {fps:.1f} fps", (8, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()

    frame_bytes = args.height * args.width * CHANNELS
    message_bytes = HEADER.size + 2 * frame_bytes  # left + right

    cv2 = None
    if args.viz:
        try:
            import cv2 as _cv2

            cv2 = _cv2
            cv2.namedWindow("ZED stereo (q/Esc to quit)", cv2.WINDOW_NORMAL)
        except ImportError:
            print("[receiver] opencv not available: disabling --viz.")
            args.viz = False

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.setsockopt(zmq.LINGER, 0)
    # In --viz mode, don't block forever on recv(): time out so the GUI stays responsive and we can
    # report that no frames are arriving (blank window == blocked on recv, not a broken display).
    if args.viz:
        subscriber.setsockopt(zmq.RCVTIMEO, 200)  # ms
    subscriber.connect(args.connect)

    print(f"[receiver] connected to {args.connect} | expecting stereo {args.height}x{args.width}x{CHANNELS}")
    if args.viz:
        print("[receiver] real-time viewer: press 'q' or Esc in the window to quit.")

    fps = 0.0  # smoothed display FPS
    last_t = None
    frames_seen = 0
    last_waiting_log = 0.0
    try:
        while True:
            if args.viz:
                result = try_receive_stereo(subscriber, args.height, args.width, frame_bytes, message_bytes)
                if result is None:
                    # No frame this interval. Keep the window alive and warn once/sec.
                    now = time.perf_counter()
                    if frames_seen == 0 and now - last_waiting_log > 1.0:
                        print(f"[receiver] waiting for frames from {args.connect} ... (none received yet)")
                        last_waiting_log = now
                    if cv2 is not None:
                        placeholder = np.zeros((360, args.viz_width, 3), np.uint8)
                        cv2.putText(
                            placeholder, "waiting for frames...", (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                        )
                        cv2.imshow("ZED stereo (q/Esc to quit)", placeholder)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                    continue
                frame_id, timestamp_ns, left, right = result
            else:
                frame_id, timestamp_ns, left, right = receive_stereo(
                    subscriber, args.height, args.width, frame_bytes, message_bytes
                )

            frames_seen += 1
            process_frames(left, right, frame_id, timestamp_ns)

            if args.viz and cv2 is not None:
                now = time.perf_counter()
                if last_t is not None:
                    dt = now - last_t
                    inst = 1.0 / dt if dt > 0 else 0.0
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                last_t = now

                canvas = merge_side_by_side(cv2, left, right, args.viz_width, fps, frame_id)
                cv2.imshow("ZED stereo (q/Esc to quit)", canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # q or Esc
                    break
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()
        context.term()
        if args.viz and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
