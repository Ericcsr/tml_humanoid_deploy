#!/usr/bin/env python3
"""
MOCK GR00T VLA server — stands in for stair-rendering/stair_vla_server.py so the
bridge can be tested without a GPU / the rendering env / GR00T weights.

Speaks the SAME Redis protocol (namespace ``vla:``, latest-wins STRING keys,
pickle) and emits the SAME heading+sincos ACTION keys the real checkpoint does:
  joint_pos_sincos(58), contact(10), root_dpos_local(3), root_dyaw(1),
  root_deheaded_rot6d(6)   -- each shaped (H, dim).

Instead of running the model, it REPLAYS a reference clip: it keeps a playhead
into the clip and, per received obs, emits the next H frames encoded as FORWARD
adjacent-frame heading deltas (the exact inverse of the bridge's
integrate_heading_chunk), then advances the playhead by (H - plan_latency). When
the bridge integrates these from the robot's live anchor and FK's them, it
reproduces the clip's motion — a faithful end-to-end exercise of the real path.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import redis
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _DEPLOY_ROOT)
sys.path.insert(0, os.path.dirname(_HERE))  # for heading_sincos
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402
from heading_sincos import (  # noqa: E402
    joints_to_sincos, rot6d_from_matrix, yaw_of_matrix, Rz, wrap_angle,
)

NUM_JOINTS = 29


def _expand_4way_to_10way(m4):
    """[Lfoot,Rfoot,Lwrist,Rwrist] -> 10-way: feet->cols0,2; wrists->cols5,7; rest 0.
    (Matches rl_policy.expand_4way_limb_to_10way_with_zero_seat.)"""
    out = np.zeros(10, np.float32)
    out[0] = m4[0]; out[2] = m4[1]; out[5] = m4[2]; out[7] = m4[3]
    return out


def _forward_deltas(P, Q_wxyz, k, H):
    """Encode clip frames [k .. k+H] as H forward heading deltas, so that
    integrate_heading_chunk(anchor=frame k) reproduces frames k+1..k+H.
      dpos[i]  = R_i^T (p_{i+1}-p_i)             (source-frame translation)
      dyaw[i]  = wrap(yaw_{i+1}-yaw_i)
      deh[i]   = rot6d(Rz(-yaw_{i+1}) R_{i+1})   (target-frame absolute tilt)
    where frame index j maps to clip index min(k+j, T-1) (clamped/held at the end).
    """
    T = P.shape[0]
    dpos = np.zeros((H, 3), np.float32)
    dyaw = np.zeros((H, 1), np.float32)
    deh = np.zeros((H, 6), np.float32)

    def RM(j):
        j = min(j, T - 1)
        return R.from_quat(Q_wxyz[j][[1, 2, 3, 0]]).as_matrix()

    def pos(j):
        return P[min(j, T - 1)]

    for i in range(H):
        Ri = RM(k + i)
        Rn = RM(k + i + 1)
        yaw_i = yaw_of_matrix(Ri)
        yaw_n = yaw_of_matrix(Rn)
        dpos[i] = (Ri.T @ (pos(k + i + 1) - pos(k + i))).astype(np.float32)
        dyaw[i] = wrap_angle(yaw_n - yaw_i)
        deh[i] = rot6d_from_matrix(Rz(-yaw_n) @ Rn)
    return dpos, dyaw, deh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True, help="Path to a *_bm_norm .npz clip to replay.")
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--redis_ns", default="vla")
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--plan_latency", type=int, default=5,
                   help="Playhead advances by (H - plan_latency) per obs (mirrors the bridge).")
    p.add_argument("--infer_ms", type=float, default=40.0, help="Simulated inference delay.")
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--no_loop", dest="loop", action="store_false")
    args = p.parse_args()

    d = np.load(args.clip)
    jp_isaac = d["joint_pos"].astype(np.float32)             # (T,29) ISAAC
    P = d["body_pos_w"][:, 0, :].astype(np.float64)          # (T,3)
    Q = d["body_quat_w"][:, 0, :].astype(np.float64)         # (T,4) wxyz
    if "contact_mask" in d.files:
        cm4 = d["contact_mask"].astype(np.float32)           # (T,4)
    else:
        cm4 = np.tile(np.array([1, 1, 0, 0], np.float32), (jp_isaac.shape[0], 1))
    T = jp_isaac.shape[0]
    H = args.horizon
    advance = max(1, H - args.plan_latency)

    ns = {k: f"{args.redis_ns}:{k}" for k in ("obs", "action", "status", "stop")}
    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    rdb.delete(ns["obs"], ns["action"], ns["stop"])
    rdb.set(ns["status"], b"ready")
    print(f"[mock_vla] ready: replaying {os.path.basename(args.clip)} T={T} H={H} "
          f"advance={advance} ns='{args.redis_ns}'", flush=True)

    chunk_id = 0
    playhead = 0
    last_obs_step = -1
    while True:
        if rdb.get(ns["stop"]) is not None:
            break
        blob = rdb.get(ns["obs"])
        if blob is None:
            time.sleep(0.001)
            continue
        obs = pickle.loads(blob)
        if obs["sim_step"] == last_obs_step:
            time.sleep(0.001)
            continue
        last_obs_step = obs["sim_step"]

        if args.infer_ms > 0:
            time.sleep(args.infer_ms / 1000.0)

        dpos, dyaw, deh = _forward_deltas(P, Q, playhead, H)
        # joints + contact for frames playhead+1 .. playhead+H (held at the end).
        idx = np.minimum(playhead + 1 + np.arange(H), T - 1)
        jsc = np.stack([joints_to_sincos(jp_isaac[j]) for j in idx]).astype(np.float32)
        contact = np.stack([_expand_4way_to_10way(cm4[j]) for j in idx]).astype(np.float32)

        action = {
            "joint_pos_sincos": jsc,           # (H,58)
            "contact": contact,                # (H,10)
            "root_dpos_local": dpos,           # (H,3)
            "root_dyaw": dyaw,                 # (H,1)
            "root_deheaded_rot6d": deh,        # (H,6)
        }
        rdb.set(ns["action"], pickle.dumps({
            "obs_step": int(obs["sim_step"]),
            "chunk_id": chunk_id,
            "infer_ms": args.infer_ms,
            "action": action,
        }, protocol=pickle.HIGHEST_PROTOCOL))
        chunk_id += 1

        playhead += advance
        if playhead >= T - 1:
            playhead = 0 if args.loop else T - 1
        if chunk_id <= 3 or chunk_id % 20 == 0:
            print(f"[mock_vla] chunk {chunk_id} for obs_step={obs['sim_step']} "
                  f"playhead={playhead}", flush=True)

    rdb.set(ns["status"], b"stopped")
    print("[mock_vla] stopped.", flush=True)


if __name__ == "__main__":
    main()
