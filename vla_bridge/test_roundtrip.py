#!/usr/bin/env python3
"""
In-process correctness test for the heading+sincos math (no Redis, no processes).

Verifies the CENTRAL invariant of the bridge: if the mock VLA server encodes a
clip's frames as FORWARD heading deltas anchored at frame k, then the bridge's
integrate_heading_chunk (anchored at the SAME frame-k world root) reproduces the
clip's absolute root trajectory and joints bit-for-bit (up to float tolerance).

Also checks: sincos encode/decode round-trip, and that FK of the reproduced
(root, joints) equals FK of the original — i.e. the vr_3point poses are correct.

Run:  python vla_bridge/test_roundtrip.py --clip <bm_norm .npz>
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _DEPLOY_ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "mocks"))

from heading_sincos import (  # noqa: E402
    joints_to_sincos, sincos_to_joints, integrate_heading_chunk,
)
from vr3_fk import VR3ForwardKinematics  # noqa: E402
from mock_vla_server import _forward_deltas  # noqa: E402
from utils.params import ISAAC_TO_MUJOCO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mujoco_xml_path", default="assets/g1/scene_29dof_flat_hand.xml")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--k", type=int, default=40, help="Anchor frame index.")
    args = ap.parse_args()

    d = np.load(args.clip)
    jp = d["joint_pos"].astype(np.float32)
    P = d["body_pos_w"][:, 0, :].astype(np.float64)
    Q = d["body_quat_w"][:, 0, :].astype(np.float64)   # wxyz
    T = jp.shape[0]
    H = args.horizon
    k = min(args.k, T - H - 2)
    fails = []

    # 1) sincos round-trip.
    err_sc = np.max(np.abs(sincos_to_joints(joints_to_sincos(jp[k])) - jp[k]))
    print(f"[test] sincos round-trip max err: {err_sc:.3e}")
    if err_sc > 1e-5:
        fails.append("sincos round-trip")

    # 2) heading delta encode -> integrate reproduces absolute root traj.
    dpos, dyaw, deh = _forward_deltas(P, Q, k, H)
    anchor_pos = P[k]
    anchor_quat_wxyz = Q[k]
    rp, rq = integrate_heading_chunk(dpos, dyaw, deh, anchor_pos, anchor_quat_wxyz)
    # Expected: frames k+1 .. k+H (clamped).
    idx = np.minimum(k + 1 + np.arange(H), T - 1)
    exp_p = P[idx]
    pos_err = np.max(np.linalg.norm(rp - exp_p, axis=1))
    # orientation error (geodesic angle), handling quat double-cover.
    from scipy.spatial.transform import Rotation as R
    q_int = R.from_quat(rq[:, [1, 2, 3, 0]])
    q_exp = R.from_quat(Q[idx][:, [1, 2, 3, 0]])
    ang_err = np.max((q_int * q_exp.inv()).magnitude())
    print(f"[test] root pos max err:  {pos_err:.3e} m")
    print(f"[test] root orn max err:  {np.degrees(ang_err):.3e} deg")
    if pos_err > 1e-3:
        fails.append(f"root pos ({pos_err:.3e})")
    if ang_err > 1e-3:
        fails.append(f"root orn ({np.degrees(ang_err):.3e} deg)")

    # 3) FK of reproduced (root, joints) == FK of original clip frames.
    fk = VR3ForwardKinematics(_DEPLOY_ROOT, args.mujoco_xml_path)
    max_fk = 0.0
    for i in range(H):
        j = idx[i]
        p_rep, _ = fk(rp[i], rq[i], jp[j][ISAAC_TO_MUJOCO])
        p_org, _ = fk(P[j], Q[j], jp[j][ISAAC_TO_MUJOCO])
        max_fk = max(max_fk, np.max(np.linalg.norm(p_rep - p_org, axis=1)))
    print(f"[test] vr3 FK max err:    {max_fk:.3e} m")
    if max_fk > 1e-3:
        fails.append(f"vr3 FK ({max_fk:.3e})")

    if fails:
        print("[test] FAIL:", "; ".join(fails))
        sys.exit(1)
    print("[test] PASS — encode->integrate->FK reproduces the clip.")


if __name__ == "__main__":
    main()
