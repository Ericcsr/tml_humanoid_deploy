#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import redis

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.redis_utils import REDIS_IP, REDIS_PORT


def _decode_message(data: bytes) -> np.ndarray:
    arr = pickle.loads(data)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def main() -> None:
    parser = ArgumentParser(description="Inspect Redis pub/sub stream for RL streaming channels.")
    parser.add_argument("--redis_ip", type=str, default=REDIS_IP, help="Redis host.")
    parser.add_argument("--redis_port", type=int, default=REDIS_PORT, help="Redis port.")
    parser.add_argument("--lower_cmd_channel", type=str, default="lower_cmd")
    parser.add_argument("--vr_pos_channel", type=str, default="vr_3point_pos_l")
    parser.add_argument("--vr_orn_channel", type=str, default="vr_3point_orn_l")
    parser.add_argument("--contact_channel", type=str, default="contact_mask")
    parser.add_argument("--anchor_pos_channel", type=str, default="motion_anchor_pos_w")
    parser.add_argument("--anchor_orn_channel", type=str, default="motion_anchor_orn_w")
    parser.add_argument("--print_every_s", type=float, default=1.0, help="Summary print interval.")
    parser.add_argument("--expect_lower_dim", type=int, default=24, help="Expected lower_cmd dim (24 or lookahead*24).")
    parser.add_argument("--expect_vr_pos_dim", type=int, default=9)
    parser.add_argument("--expect_vr_orn_dim", type=int, default=12)
    parser.add_argument("--expect_contact_dim", type=int, default=4)
    parser.add_argument("--expect_anchor_pos_dim", type=int, default=3)
    parser.add_argument("--expect_anchor_orn_dim", type=int, default=4)
    args = parser.parse_args()

    channels = {
        "lower_cmd": args.lower_cmd_channel,
        "vr_3point_pos_l": args.vr_pos_channel,
        "vr_3point_orn_l": args.vr_orn_channel,
        "contact_mask": args.contact_channel,
        "motion_anchor_pos_w": args.anchor_pos_channel,
        "motion_anchor_orn_w": args.anchor_orn_channel,
    }
    expected_dims = {
        "lower_cmd": int(args.expect_lower_dim),
        "vr_3point_pos_l": int(args.expect_vr_pos_dim),
        "vr_3point_orn_l": int(args.expect_vr_orn_dim),
        "contact_mask": int(args.expect_contact_dim),
        "motion_anchor_pos_w": int(args.expect_anchor_pos_dim),
        "motion_anchor_orn_w": int(args.expect_anchor_orn_dim),
    }

    r = redis.Redis(host=args.redis_ip, port=args.redis_port, db=0)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(*channels.values())
    ch_to_name = {v: k for k, v in channels.items()}

    counts = {k: 0 for k in channels}
    bad_shape = {k: 0 for k in channels}
    decode_err = {k: 0 for k in channels}
    decode_err_last = {k: None for k in channels}
    last_shape = {k: None for k in channels}
    last_preview = {k: None for k in channels}
    last_msg_t = {k: None for k in channels}

    print(
        f"[stream_subscriber] listening on redis {args.redis_ip}:{args.redis_port} channels: "
        f"{channels['lower_cmd']}, {channels['vr_3point_pos_l']}, {channels['vr_3point_orn_l']}, {channels['contact_mask']}",
        flush=True,
    )
    print(f"[stream_subscriber] expected dims: {expected_dims}", flush=True)
    last_print_t = time.monotonic()

    try:
        while True:
            msg = pubsub.get_message(timeout=0.1)
            if msg is not None and msg.get("type") == "message":
                channel = msg["channel"]
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8")
                name = ch_to_name.get(channel)
                if name is not None:
                    data = msg["data"]
                    if isinstance(data, (bytes, bytearray)):
                        try:
                            arr = _decode_message(data)
                            counts[name] += 1
                            last_shape[name] = int(arr.size)
                            last_preview[name] = arr[: min(4, arr.size)].tolist()
                            last_msg_t[name] = time.monotonic()
                            if arr.size != expected_dims[name]:
                                bad_shape[name] += 1
                        except Exception as exc:
                            decode_err[name] += 1
                            if decode_err_last[name] is None:
                                preview = bytes(data[:24]).hex() if len(data) > 0 else ""
                                decode_err_last[name] = (
                                    f"{type(exc).__name__}: {exc} "
                                    f"(raw_len={len(data)}, raw_hex_prefix={preview})"
                                )
                    else:
                        decode_err[name] += 1
                        if decode_err_last[name] is None:
                            decode_err_last[name] = f"Non-bytes payload type: {type(data)} value={data}"

            now = time.monotonic()
            if now - last_print_t >= max(0.1, float(args.print_every_s)):
                lines = ["[stream_subscriber] ----"]
                for name in (
                    "lower_cmd",
                    "vr_3point_pos_l",
                    "vr_3point_orn_l",
                    "contact_mask",
                    "motion_anchor_pos_w",
                    "motion_anchor_orn_w",
                ):
                    age = None if last_msg_t[name] is None else now - float(last_msg_t[name])
                    lines.append(
                        f"  {name:16s} count={counts[name]:6d} "
                        f"shape={last_shape[name]} bad_shape={bad_shape[name]} decode_err={decode_err[name]} "
                        f"last_age_s={None if age is None else round(age, 3)} preview={last_preview[name]}"
                    )
                    if decode_err_last[name] is not None:
                        lines.append(f"    decode_err_last={decode_err_last[name]}")
                print("\n".join(lines), flush=True)
                last_print_t = now
    except KeyboardInterrupt:
        print("[stream_subscriber] stopped.", flush=True)


if __name__ == "__main__":
    main()
