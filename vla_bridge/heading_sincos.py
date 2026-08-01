"""
Root/joint math for the GR00T "heading + sin/cos" representation
(``vlk_g1_contact_fk_heading_sincos_pelvis_config``).

These functions are ported VERBATIM (same formulas, same conventions) from the
SYNC path of ``stair-rendering/stair_sim_eval.py`` — the async path there raises
NotImplementedError for the heading/sincos variant, so the reference math lives
only in the sync loop. This module re-implements it stand-alone so the deploy-side
bridge (``vla_motion_bridge.py``) can decode/integrate GR00T chunks and pack obs
without importing Isaac / the sim harness.

Conventions (must match the harness exactly):
  * Stored quaternions are **wxyz**. scipy always gets **xyzw** (``[[1,2,3,0]]``)
    and returns wxyz (``[[3,0,1,2]]``).
  * ``joint_pos_sincos`` layout is CONTIGUOUS BLOCKS ``[sin(q_0..28), cos(q_0..28)]``
    (58 dims), NOT interleaved. Joint order is ISAAC.
  * ``root_deheaded_rot6d`` is the ABSOLUTE de-headed tilt (roll/pitch with the
    heading removed), not a delta, in both STATE and ACTION.
  * ``root_dpos_local`` / ``root_dyaw`` are backward deltas over one control step.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ── 6D-rotation helpers (Zhou et al. continuous representation) ──
def rot6d_from_matrix(Rm):
    """(3,3) -> (6,): first two columns of the rotation matrix, stacked."""
    return np.concatenate([Rm[:, 0], Rm[:, 1]]).astype(np.float32)


def matrix_from_rot6d(x):
    """(6,) -> (3,3) via Gram-Schmidt (inverse of rot6d_from_matrix)."""
    a1 = np.asarray(x[:3], np.float64)
    a2 = np.asarray(x[3:6], np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2p = a2 - (b1 @ a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def yaw_of_matrix(Rm):
    """Heading (yaw about +Z) of a rotation matrix: atan2(R[1,0], R[0,0])."""
    return float(np.arctan2(Rm[1, 0], Rm[0, 0]))


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── sin/cos joint encode / decode (ISAAC order, [sin(29), cos(29)] blocks) ──
def joints_to_sincos(joints_isaac):
    """(29,) joint angles -> (58,) [sin(q), cos(q)] contiguous blocks."""
    j = np.asarray(joints_isaac, np.float64)
    return np.concatenate([np.sin(j), np.cos(j)]).astype(np.float32)


def sincos_to_joints(sincos):
    """(..., 58) -> (..., 29): q = atan2(sin_block, cos_block)."""
    sc = np.asarray(sincos, np.float64)
    nj = sc.shape[-1] // 2
    return np.arctan2(sc[..., :nj], sc[..., nj:]).astype(np.float32)


# ── OBS side: backward delta of `now` w.r.t. `prev` (heading variant) ──
def heading_backward_delta(p_now, q_now_wxyz, p_prev, q_prev_wxyz):
    """Return (root_dpos_local (3,), root_dyaw (1,), root_deheaded_rot6d (6,)).

    dpos_local = R_prev^T (p_now - p_prev); dyaw = wrap(yaw_now - yaw_prev);
    deheaded   = rot6d(Rz(-yaw_now) @ R_now)  (ABSOLUTE tilt at the current frame).
    On the first obs (p_prev is None): dpos=0, dyaw=0, deheaded from R_now.
    """
    R_now = R.from_quat(np.asarray(q_now_wxyz)[[1, 2, 3, 0]]).as_matrix()
    yaw_now = yaw_of_matrix(R_now)
    deheaded = rot6d_from_matrix(Rz(-yaw_now) @ R_now)
    if p_prev is None:
        return np.zeros(3, np.float32), np.zeros(1, np.float32), deheaded
    R_prev = R.from_quat(np.asarray(q_prev_wxyz)[[1, 2, 3, 0]]).as_matrix()
    dpos = (R_prev.T @ (np.asarray(p_now) - np.asarray(p_prev))).astype(np.float32)
    dyaw = np.array([wrap_angle(yaw_now - yaw_of_matrix(R_prev))], np.float32)
    return dpos, dyaw, deheaded


# ── ACTION side: integrate a chunk into an absolute root trajectory ──
def integrate_heading_chunk(dpos_local, dyaw, deheaded_rot6d,
                            anchor_root_pos, anchor_root_quat_wxyz):
    """Integrate root_dpos_local + root_dyaw + ABSOLUTE root_deheaded_rot6d from an
    anchor into absolute (root_pos (H,3), root_quat (H,4) wxyz).

    Per frame i: R_src = Rz(yaw_run) @ r6toR(src_deh); p += R_src @ dpos[i];
    yaw_run += dyaw[i]; R_next = Rz(yaw_run) @ r6toR(deh[i]).
    out[i] is the pose AFTER applying delta i (one frame ahead of the anchor).
    Only yaw integrates -> no roll/pitch drift. Anchor should be the robot's
    CURRENT world root (re-anchored fresh per chunk).
    """
    dp = np.asarray(dpos_local, np.float64)              # (H,3)
    dy = np.asarray(dyaw, np.float64).reshape(-1)         # (H,)
    deh = np.asarray(deheaded_rot6d, np.float64)          # (H,6)
    H = dp.shape[0]
    p = np.asarray(anchor_root_pos, np.float64).copy()
    R_anchor = R.from_quat(np.asarray(anchor_root_quat_wxyz, np.float64)[[1, 2, 3, 0]]).as_matrix()
    yaw_run = yaw_of_matrix(R_anchor)
    out_p = np.empty((H, 3), np.float32)
    out_q = np.empty((H, 4), np.float32)
    # Source-frame tilt for the translation: anchor's de-headed tilt at i=0,
    # else the previous target tilt.
    src_deh = rot6d_from_matrix(Rz(-yaw_run) @ R_anchor)
    for i in range(H):
        R_src = Rz(yaw_run) @ matrix_from_rot6d(src_deh)
        p = p + R_src @ dp[i]
        yaw_run = yaw_run + float(dy[i])
        R_next = Rz(yaw_run) @ matrix_from_rot6d(deh[i])
        out_p[i] = p.astype(np.float32)
        out_q[i] = R.from_matrix(R_next).as_quat()[[3, 0, 1, 2]].astype(np.float32)
        src_deh = deh[i]
    return out_p, out_q
