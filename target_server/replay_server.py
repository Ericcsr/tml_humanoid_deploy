#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import redis
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.params import ISAAC_TO_MUJOCO
from utils.redis_utils import REDIS_IP, REDIS_PORT


VR_3POINT_INDICES = [28, 29, 9]
VR_3POINT_OFFSETS = np.array(
    [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
    dtype=np.float32,
)


def build_stream_frames(ref_motion_path, contact_labels_path):
    ref_motion = np.load(ref_motion_path)
    motion_length = ref_motion["joint_pos"].shape[0]
    lower_joint_indices = ISAAC_TO_MUJOCO[:12]

    ref_q_pos = ref_motion["joint_pos"].astype(np.float32)
    ref_q_vel = ref_motion["joint_vel"].astype(np.float32)
    ref_anchor_pos = ref_motion["body_pos_w"][:, 0].astype(np.float32)
    ref_anchor_orn = ref_motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]].astype(np.float32)
    ref_vr_3point_pos = ref_motion["body_pos_w"][:, VR_3POINT_INDICES].astype(np.float32)
    ref_vr_3point_orn = ref_motion["body_quat_w"][:, VR_3POINT_INDICES][:, :, [1, 2, 3, 0]].astype(np.float32)

    ref_vr_3point_pos[:, 0] += Rotation.from_quat(ref_vr_3point_orn[:, 0]).apply(VR_3POINT_OFFSETS[None, 0, :])
    ref_vr_3point_pos[:, 1] += Rotation.from_quat(ref_vr_3point_orn[:, 1]).apply(VR_3POINT_OFFSETS[None, 1, :])
    ref_vr_3point_pos[:, 2] += Rotation.from_quat(ref_vr_3point_orn[:, 2]).apply(VR_3POINT_OFFSETS[None, 2, :])

    use_contact_labels = bool(contact_labels_path and str(contact_labels_path).strip())
    if use_contact_labels and str(contact_labels_path).strip().lower() != "default":
        contact_labels = np.load(contact_labels_path, allow_pickle=True).item()
        contact_mask = np.asarray(contact_labels["contact_mask"], dtype=np.float32)
        if contact_mask.shape[0] < motion_length:
            pad = np.zeros((motion_length - contact_mask.shape[0], contact_mask.shape[1]), dtype=np.float32)
            contact_mask = np.vstack([contact_mask, pad])
        contact_mask = contact_mask[:motion_length]
    else:
        contact_mask = np.zeros((motion_length, 4), dtype=np.float32)

    frames = []
    for i in range(motion_length):
        lower_cmd = np.hstack(
            [
                ref_q_pos[i, lower_joint_indices],
                ref_q_vel[i, lower_joint_indices],
            ]
        ).astype(np.float32)
        vr_3point_pos_l = Rotation.from_quat(ref_anchor_orn[i]).inv().apply(
            ref_vr_3point_pos[i] - ref_anchor_pos[i][None, :]
        )
        vr_3point_orn_l = (
            Rotation.from_quat(ref_anchor_orn[i]).inv() * Rotation.from_quat(ref_vr_3point_orn[i])
        ).as_quat(scalar_first=True)
        frames.append(
            {
                "lower_cmd": lower_cmd,
                "vr_3point_pos_l": vr_3point_pos_l.reshape(-1).astype(np.float32),
                "vr_3point_orn_l": vr_3point_orn_l.reshape(-1).astype(np.float32),
                "contact_mask": contact_mask[i].reshape(-1).astype(np.float32),
                "motion_anchor_pos_w": ref_anchor_pos[i].reshape(-1).astype(np.float32),
                "motion_anchor_orn_w": ref_anchor_orn[i].reshape(-1).astype(np.float32),  # xyzw
            }
        )
    return frames


def main():
    parser = ArgumentParser(description="Replay reference motion to Redis topics at fixed rate.")
    parser.add_argument("--ref_motion_path", type=str, required=True, help="Path to .npz reference motion.")
    parser.add_argument(
        "--contact_labels_path",
        type=str,
        default="default",
        help='Path to contact labels .npy (dict with "contact_mask"), or "default"/empty for zero mask.',
    )
    parser.add_argument("--redis_ip", type=str, default=REDIS_IP, help="Redis host.")
    parser.add_argument("--redis_port", type=int, default=REDIS_PORT, help="Redis port.")
    parser.add_argument("--hz", type=float, default=50.0, help="Publish rate in Hz.")
    parser.add_argument("--loop", action="store_true", help="Loop playback forever.")
    parser.add_argument("--lower_cmd_channel", type=str, default="lower_cmd")
    parser.add_argument("--vr_pos_channel", type=str, default="vr_3point_pos_l")
    parser.add_argument("--vr_orn_channel", type=str, default="vr_3point_orn_l")
    parser.add_argument("--contact_channel", type=str, default="contact_mask")
    parser.add_argument("--anchor_pos_channel", type=str, default="motion_anchor_pos_w")
    parser.add_argument("--anchor_orn_channel", type=str, default="motion_anchor_orn_w")
    args = parser.parse_args()

    if args.hz <= 0:
        raise ValueError("--hz must be positive.")

    frames = build_stream_frames(args.ref_motion_path, args.contact_labels_path)
    redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, db=0)

    period = 1.0 / args.hz
    frame_idx = 0
    next_tick = time.perf_counter()
    print(
        "[replay_server] Streaming frames:",
        len(frames),
        f"rate={args.hz:.2f}Hz loop={args.loop}",
        flush=True,
    )
    print(
        "[replay_server] Channels:",
        args.lower_cmd_channel,
        args.vr_pos_channel,
        args.vr_orn_channel,
        args.contact_channel,
        args.anchor_pos_channel,
        args.anchor_orn_channel,
        flush=True,
    )

    while True:
        frame = frames[frame_idx]
        redis_client.publish(args.lower_cmd_channel, pickle.dumps(frame["lower_cmd"], protocol=pickle.HIGHEST_PROTOCOL))
        redis_client.publish(
            args.vr_pos_channel,
            pickle.dumps(frame["vr_3point_pos_l"], protocol=pickle.HIGHEST_PROTOCOL),
        )
        redis_client.publish(
            args.vr_orn_channel,
            pickle.dumps(frame["vr_3point_orn_l"], protocol=pickle.HIGHEST_PROTOCOL),
        )
        redis_client.publish(
            args.contact_channel,
            pickle.dumps(frame["contact_mask"], protocol=pickle.HIGHEST_PROTOCOL),
        )
        redis_client.publish(
            args.anchor_pos_channel,
            pickle.dumps(frame["motion_anchor_pos_w"], protocol=pickle.HIGHEST_PROTOCOL),
        )
        redis_client.publish(
            args.anchor_orn_channel,
            pickle.dumps(frame["motion_anchor_orn_w"], protocol=pickle.HIGHEST_PROTOCOL),
        )

        frame_idx += 1
        if frame_idx >= len(frames):
            if not args.loop:
                break
            frame_idx = 0

        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()

    print("[replay_server] Done.", flush=True)


if __name__ == "__main__":
    main()
