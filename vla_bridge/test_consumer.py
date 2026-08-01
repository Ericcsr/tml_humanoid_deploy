#!/usr/bin/env python3
"""
End-to-end consumer test: drives the REAL RLStreamingContactPolicy against the
live bridge output, exactly as run_controller.py would (minus the robot).

Assumes the mock harness (mock_robot_state + mock_camera + mock_vla_server) and
vla_motion_bridge.py are already running. This process:
  * builds RLStreamingContactPolicy from the streaming config (loads the tracking
    ONNX),
  * each control step: reads root_data (as the controller does), fabricates a
    G1RobotState with live root pose + zero proprio, calls prepare_control_signals
    -> prepare_obs -> get_action,
  * asserts the policy consumed real streamed values (non-default anchor, moving)
    and produced a finite 29-dim action.

Run (while the harness is up):
  python vla_bridge/test_consumer.py --config exported_policies/stair_assets/experiment_stair_streaming.yaml
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import redis
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DEPLOY_ROOT)

from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402
from utils.robot_states import G1RobotState  # noqa: E402
from rl_policy import RLStreamingContactPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exported_policies/stair_assets/experiment_stair_streaming.yaml")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--control_dt", type=float, default=0.02)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["rl_policy"]

    policy = RLStreamingContactPolicy(
        cfg["onnx_model_path"], cfg["obs_names"],
        use_sim=False,
        lookahead_steps=cfg.get("lookahead_steps", 1),
        lookahead_frame_skips=cfg.get("lookahead_frame_skips", 1),
        hist_names=cfg.get("history_names", []),
        hist_length=cfg.get("history_length", 1),
        redis_ip=cfg.get("streaming_redis_ip", REDIS_IP),
        redis_port=cfg.get("streaming_redis_port", REDIS_PORT),
        default_contact_label=cfg.get("default_contact_label", None),
        use_10way_contact=cfg.get("use_10way_contact", False),
        use_5dim_contact_from_4dim=cfg.get("use_5dim_contact_from_4dim", False),
        ref_motion_start_index=0,
    )
    rdb = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=0)

    rs = G1RobotState()
    rs.q = np.zeros(29, np.float32)
    rs.dq = np.zeros(29, np.float32)
    rs.omega = np.zeros(3, np.float32)
    rs.last_action = np.zeros(29, np.float32)

    seen_anchor = set()
    n_ok = 0
    for t in range(args.steps):
        rd = rdb.get("root_data")
        if rd is not None:
            rd = np.asarray(pickle.loads(rd), np.float32)
            rs.root_pos = rd[:3]
            rs.root_orn = rd[3:7]        # xyzw
            rs.root_vel = rd[7:10]
        cs = policy.prepare_control_signals(rs)
        obs = policy.prepare_obs(rs, cs)
        action = policy.get_action(obs, start_ticker=True)
        assert action.shape[-1] == 29, action.shape
        assert np.all(np.isfinite(action)), "non-finite action"
        seen_anchor.add(tuple(np.round(policy.latest_motion_anchor_pos_w, 3)))
        n_ok += 1
        time.sleep(args.control_dt)

    print(f"[test_consumer] steps ok: {n_ok}/{args.steps}")
    print(f"[test_consumer] received_any_stream: {policy.received_any_stream}")
    print(f"[test_consumer] distinct anchor positions seen: {len(seen_anchor)}")
    print(f"[test_consumer] last anchor_pos_w: {policy.latest_motion_anchor_pos_w}")
    print(f"[test_consumer] last contact_mask(10): {np.round(policy.latest_contact_mask,2)}")
    ok = (policy.received_any_stream and len(seen_anchor) > 5)
    if not ok:
        print("[test_consumer] FAIL — policy did not consume a moving live stream.")
        sys.exit(1)
    print("[test_consumer] PASS — RLStreamingContactPolicy tracks the bridge stream.")


if __name__ == "__main__":
    main()
