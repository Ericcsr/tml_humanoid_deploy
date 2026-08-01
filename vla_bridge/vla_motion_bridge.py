#!/usr/bin/env python3
"""
VLA motion bridge — feeds VLA (GR00T) reference motion to run_controller.py.

Sits between the GR00T inference server (``stair-rendering/stair_vla_server.py``,
``rendering`` env) and the tracking controller (``run_controller.py`` with
``use_streaming_motion: True``), speaking Redis on both sides. Runs in the ``bm``
env (needs only numpy/scipy/mujoco/redis/cv2 — no Isaac, no Unitree SDK).

Target checkpoint: the HEADING + SIN/COS pelvis variant
(``vlk_g1_contact_fk_heading_sincos_pelvis_config`` /
``vlk_g1_v1_heading_sincos/vlk_apartment_noforce_nofk_lowres_gma_p``):
  STATE  = joint_pos_sincos(58), joint_vel(29), root_dpos_local(3), root_dyaw(1),
           root_deheaded_rot6d(6), contact(10) + pelvis_left/right_view
  ACTION = joint_pos_sincos(58), contact(10), root_dpos_local(3), root_dyaw(1),
           root_deheaded_rot6d(6)   over a 25-step horizon.

Data flow (all Redis @ localhost:6379, db 0)::

    run_state_estimation_slam_only.py --use_imu   (needs superodometry; not here)
        proprio_data  ─▶  root_data
                           │
    run_controller.py --use_odom  writes proprio_data, reads root_data,
        RLStreamingContactPolicy ◀── 6 pub/sub channels ── THIS BRIDGE
                           │
    THIS BRIDGE                                   stair_vla_server.py (GR00T)
      reads  root_data (world root pose, xyzw)     ◀── vla:action  (chunk)
      reads  proprio_data (q,dq MuJoCo)            ──▶ vla:obs      (state+imgs)
      reads  camera:<view> (JPEG, latest-wins)
      per VLA chunk: decode sin/cos -> joints; integrate heading deltas from the
        robot's CURRENT root_data anchor -> absolute root traj; MuJoCo-FK the
        3-point wrist/torso poses; then STREAM the chunk frame-by-frame at 50 Hz
        onto the 6 controller channels, re-planning near the end of each chunk.

The 6 output channels + payloads exactly match ``target_server/replay_server.py``
(the file-based reference producer this bridge replaces):
  lower_cmd(24)  vr_3point_pos_l(9)  vr_3point_orn_l(12, wxyz)
  contact_mask(10)  motion_anchor_pos_w(3)  motion_anchor_orn_w(4, xyzw)
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
_DEPLOY_ROOT = os.path.dirname(_HERE)
if _DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _DEPLOY_ROOT)

from utils.params import ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC  # noqa: E402
from utils.redis_utils import REDIS_IP, REDIS_PORT  # noqa: E402

from heading_sincos import (  # noqa: E402
    joints_to_sincos,
    sincos_to_joints,
    heading_backward_delta,
    integrate_heading_chunk,
)
from vr3_fk import VR3ForwardKinematics  # noqa: E402

NUM_JOINTS = 29
CONTACT_DIM = 10           # GR00T native 10-way; streaming policy keeps [:8]+zeros(2)
LOWER_JOINT_INDICES = ISAAC_TO_MUJOCO[:12]   # 12 lower-body joints (matches replay_server)

# The 6 controller channels (names == replay_server defaults).
CH_LOWER = "lower_cmd"
CH_VR_POS = "vr_3point_pos_l"
CH_VR_ORN = "vr_3point_orn_l"
CH_CONTACT = "contact_mask"
CH_ANCHOR_POS = "motion_anchor_pos_w"
CH_ANCHOR_ORN = "motion_anchor_orn_w"


def _default_contact_10():
    """Both feet down, everything else off -> 10-way with zero seat/obj columns.
    4-way [Lfoot,Rfoot,0,0] mapped to 10-way cols [0,2] (env), rest 0."""
    c = np.zeros(CONTACT_DIM, np.float32)
    c[0] = 1.0   # Lfoot env
    c[2] = 1.0   # Rfoot env
    return c


class VLAMotionBridge:
    def __init__(self, args):
        self.args = args
        self.rdb = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        self.vla = {
            "obs": f"{args.redis_ns}:obs",
            "action": f"{args.redis_ns}:action",
            "status": f"{args.redis_ns}:status",
            "stop": f"{args.redis_ns}:stop",
        }
        # video_key -> camera redis topic (latest-wins STRING holding JPEG bytes).
        self.video_topics = dict(args.video_topics)
        self.fk = VR3ForwardKinematics(args.deploy_root, args.mujoco_xml_path)

        self.control_dt = args.control_dt
        self.motion_fps = args.motion_fps
        self.horizon = args.horizon
        self.plan_latency = int(np.clip(args.plan_latency, 0, self.horizon - 1))

        # 2-deep root history for a 1-control-step backward delta (matches sim_eval).
        self._last_root = {"pos": None, "quat": None}   # quat wxyz
        self._prev_root = {"pos": None, "quat": None}

        # Current chunk (streamed frame-by-frame onto the channels).
        self._chunk = None          # dict of per-frame ref arrays
        self._chunk_pos = 0
        self._chunk_len = 0
        self._pending = False       # obs sent, awaiting a fresh chunk
        self._held_chunk_id = -1
        self._obs_step = 0
        self._anchor_at_obs = None  # (pos, quat_wxyz) captured when obs was sent
        self._last_contact = _default_contact_10()
        self._starved = 0

    # ── Redis reads ──
    def _get_pickle(self, key):
        blob = self.rdb.get(key)
        return None if blob is None else pickle.loads(blob)

    def _read_root(self):
        """root_data = [root_pos(3) | root_orn(4, xyzw) | root_vel(3)]. Returns
        (pos(3), quat_wxyz(4)) or None."""
        d = self._get_pickle("root_data")
        if d is None:
            return None
        d = np.asarray(d, np.float64)
        pos = d[:3].copy()
        quat_wxyz = d[3:7][[3, 0, 1, 2]].copy()   # xyzw -> wxyz
        return pos, quat_wxyz

    def _read_proprio(self):
        """proprio_data = [q(29) | dq(29) | omega(3) | imu_quat(4, xyzw)] (MuJoCo
        joint order). Returns (q_isaac(29), dq_isaac(29)) or None."""
        d = self._get_pickle("proprio_data")
        if d is None:
            return None
        d = np.asarray(d, np.float32)
        q_mj = d[:NUM_JOINTS]
        dq_mj = d[NUM_JOINTS:2 * NUM_JOINTS]
        return q_mj[MUJOCO_TO_ISAAC].copy(), dq_mj[MUJOCO_TO_ISAAC].copy()

    def _read_images(self):
        """Read the latest JPEG bytes for each video key. Returns {video_key: bytes}
        or None if any camera topic is empty."""
        out = {}
        for vkey, topic in self.video_topics.items():
            blob = self.rdb.get(topic)
            if blob is None:
                return None
            out[vkey] = bytes(blob)
        return out

    # ── VLA obs / action ──
    def _wait_for_server(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.rdb.get(self.vla["status"]) == b"ready":
                return True
            time.sleep(0.2)
        return False

    def _publish_obs(self, q_isaac, dq_isaac, root_now, images):
        """Pack + publish vla:obs. root_now = (pos, quat_wxyz). Records the anchor
        used for integrating the resulting chunk."""
        p_prev = self._prev_root["pos"]
        q_prev = self._prev_root["quat"]
        dpos, dyaw, deh = heading_backward_delta(
            root_now[0], root_now[1], p_prev, q_prev
        )
        state = {
            "joint_pos_sincos": joints_to_sincos(q_isaac),
            "joint_vel": np.asarray(dq_isaac, np.float32),
            "root_dpos_local": dpos,
            "root_dyaw": dyaw,
            "root_deheaded_rot6d": deh,
            "contact": self._last_contact.copy(),
        }
        # Serialize state arrays as PLAIN PYTHON LISTS (not numpy arrays). The VLA
        # server runs in a different conda env whose numpy may differ from ours
        # (e.g. rendering=1.26 vs bm=2.x); a numpy-2 pickle references numpy._core,
        # which numpy<2 cannot import -> the server would fail to unpickle EVERY obs
        # and never emit a chunk. Lists are version-agnostic; the server rebuilds
        # them via np.asarray(state[k], np.float32). Images are raw JPEG bytes and
        # cross versions fine.
        msg = {
            "sim_step": int(self._obs_step),
            "state": {k: np.asarray(v, np.float32).tolist() for k, v in state.items()},
            "images": images,
            "task": self.args.task,
            "mode": "sim_reset",
            "contact": None,          # contact already in state
        }
        self.rdb.set(self.vla["obs"], pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
        self._anchor_at_obs = (np.asarray(root_now[0], np.float64).copy(),
                               np.asarray(root_now[1], np.float64).copy())
        self._obs_step += 1

    def _poll_action(self):
        """Return a freshly-integrated chunk dict, or None if no NEW chunk yet."""
        d = self._get_pickle(self.vla["action"])
        if d is None or d.get("chunk_id") == self._held_chunk_id:
            return None
        self._held_chunk_id = d["chunk_id"]
        act = d["action"]
        return self._build_chunk(act)

    def _build_chunk(self, act):
        """Decode sin/cos + integrate heading deltas + FK the 3-point poses, then
        precompute every per-frame channel payload. Anchored at the obs-time root."""
        jp = sincos_to_joints(act["joint_pos_sincos"]).astype(np.float32)  # (H,29) ISAAC
        H = jp.shape[0]
        anchor = self._anchor_at_obs
        if anchor is None:   # first chunk before any obs anchor: use current root
            root = self._read_root()
            anchor = (root[0], root[1]) if root is not None else (np.zeros(3), np.array([1.0, 0, 0, 0]))
        rp, rq_wxyz = integrate_heading_chunk(
            act["root_dpos_local"], act["root_dyaw"], act["root_deheaded_rot6d"],
            anchor[0], anchor[1],
        )
        rq_xyzw = rq_wxyz[:, [1, 2, 3, 0]]
        contact = np.asarray(act.get("contact", np.tile(_default_contact_10(), (H, 1))), np.float32)
        if contact.shape[1] != CONTACT_DIM:
            raise ValueError(f"chunk contact dim {contact.shape[1]} != {CONTACT_DIM}")

        # Reference joint velocity: finite-difference at motion_fps (matches sim_eval).
        jv = np.zeros_like(jp)
        if H >= 2:
            jv[:-1] = (jp[1:] - jp[:-1]) * float(self.motion_fps)
            jv[-1] = jv[-2]

        # FK the 3-point (wrist/torso) world poses per frame, then anchor-local.
        vr_pos_l = np.zeros((H, 9), np.float32)
        vr_orn_l = np.zeros((H, 12), np.float32)
        lower = np.zeros((H, 24), np.float32)
        for i in range(H):
            poses_w, orns_w = self.fk(rp[i], rq_wxyz[i], jp[i][ISAAC_TO_MUJOCO])  # (3,3),(3,4 xyzw)
            anchor_inv = R.from_quat(rq_xyzw[i]).inv()
            pl = anchor_inv.apply(poses_w - rp[i][None, :])                       # (3,3)
            ol = (anchor_inv * R.from_quat(orns_w)).as_quat(scalar_first=True)    # (3,4) wxyz
            vr_pos_l[i] = pl.reshape(-1)
            vr_orn_l[i] = ol.reshape(-1)
            lower[i] = np.hstack([jp[i, LOWER_JOINT_INDICES], jv[i, LOWER_JOINT_INDICES]])

        return {
            "lower_cmd": lower,               # (H,24)
            "vr_pos_l": vr_pos_l,             # (H,9)
            "vr_orn_l": vr_orn_l,             # (H,12) wxyz
            "contact": contact,               # (H,10)
            "anchor_pos_w": rp.astype(np.float32),        # (H,3)
            "anchor_orn_w": rq_xyzw.astype(np.float32),   # (H,4) xyzw
            "H": H,
        }

    # ── channel emit ──
    def _publish_frame(self, chunk, i):
        pub = self.rdb.publish
        P = pickle.HIGHEST_PROTOCOL
        pub(CH_LOWER, pickle.dumps(chunk["lower_cmd"][i], protocol=P))
        pub(CH_VR_POS, pickle.dumps(chunk["vr_pos_l"][i], protocol=P))
        pub(CH_VR_ORN, pickle.dumps(chunk["vr_orn_l"][i], protocol=P))
        pub(CH_CONTACT, pickle.dumps(chunk["contact"][i], protocol=P))
        pub(CH_ANCHOR_POS, pickle.dumps(chunk["anchor_pos_w"][i], protocol=P))
        pub(CH_ANCHOR_ORN, pickle.dumps(chunk["anchor_orn_w"][i], protocol=P))
        self._last_contact = (chunk["contact"][i] > 0.5).astype(np.float32)

    # ── main loop ──
    def run(self):
        a = self.args
        print(f"[bridge] waiting for VLA server '{a.redis_ns}' @ "
              f"{a.redis_host}:{a.redis_port} ...", flush=True)
        if not self._wait_for_server(a.server_timeout):
            print(f"[bridge] VLA server not ready after {a.server_timeout}s; exiting.", flush=True)
            return
        print("[bridge] VLA server ready.", flush=True)

        # Block until we have a robot root + proprio + camera frames.
        print("[bridge] waiting for root_data / proprio_data / camera topics ...", flush=True)
        while True:
            root = self._read_root(); prop = self._read_proprio(); imgs = self._read_images()
            if root is not None and prop is not None and imgs is not None:
                break
            if self.rdb.get(self.vla["stop"]) is not None:
                return
            time.sleep(0.05)
        self._last_root["pos"], self._last_root["quat"] = root[0], root[1]

        # Prime: send the first obs and block for the first chunk.
        q_isaac, dq_isaac = prop
        self._publish_obs(q_isaac, dq_isaac, root, imgs)
        print("[bridge] first obs sent; waiting for first chunk ...", flush=True)
        while self._chunk is None:
            c = self._poll_action()
            if c is not None:
                self._chunk, self._chunk_len, self._chunk_pos = c, c["H"], 0
            if self.rdb.get(self.vla["stop"]) is not None:
                return
            time.sleep(0.002)
        print(f"[bridge] streaming reference motion @ {1.0/self.control_dt:.0f} Hz "
              f"(H={self.horizon}, replan_at={self.horizon - self.plan_latency}).", flush=True)

        replan_at = self.horizon - self.plan_latency
        next_tick = time.perf_counter()
        n_frames = 0
        while True:
            if self.rdb.get(self.vla["stop"]) is not None:
                print("[bridge] stop signal; exiting.", flush=True)
                break

            # Shift the 1-step root history from the freshest root_data.
            root = self._read_root()
            if root is not None:
                self._prev_root["pos"], self._prev_root["quat"] = (
                    self._last_root["pos"], self._last_root["quat"])
                self._last_root["pos"], self._last_root["quat"] = root[0], root[1]

            # Emit the current chunk frame (hold last frame if we ran off the end).
            idx = min(self._chunk_pos, self._chunk_len - 1)
            if self._chunk_pos >= self._chunk_len:
                self._starved += 1
            self._publish_frame(self._chunk, idx)
            n_frames += 1

            # Near the end of the chunk: request the next one.
            if self._chunk_pos >= replan_at and not self._pending:
                prop = self._read_proprio(); imgs = self._read_images()
                if prop is not None and imgs is not None and self._last_root["pos"] is not None:
                    self._publish_obs(prop[0], prop[1],
                                      (self._last_root["pos"], self._last_root["quat"]), imgs)
                    self._pending = True

            # New chunk arrived: swap in, resuming L frames ahead (they're now past).
            if self._pending:
                c = self._poll_action()
                if c is not None:
                    self._chunk, self._chunk_len = c, c["H"]
                    self._chunk_pos = self.plan_latency
                    self._pending = False
                else:
                    self._chunk_pos += 1
            else:
                self._chunk_pos += 1

            if n_frames % 250 == 0:
                print(f"[bridge] frames={n_frames} chunk_id={self._held_chunk_id} "
                      f"pos={self._chunk_pos}/{self._chunk_len} starved={self._starved}", flush=True)

            next_tick += self.control_dt
            dt = next_tick - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_tick = time.perf_counter()


def _parse_video_topics(items):
    """--video_topic pelvis_left_view=camera:pelvis_left_view ... -> list of pairs."""
    out = []
    for it in items:
        if "=" not in it:
            raise argparse.ArgumentTypeError(f"--video_topic must be KEY=TOPIC, got {it!r}")
        k, v = it.split("=", 1)
        out.append((k, v))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("Data flow")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redis_host", default=REDIS_IP)
    p.add_argument("--redis_port", type=int, default=REDIS_PORT)
    p.add_argument("--redis_ns", default="vla", help="VLA server key namespace.")
    p.add_argument("--deploy_root", default=_DEPLOY_ROOT,
                   help="Deploy root containing assets/g1 (for the FK scene).")
    p.add_argument("--mujoco_xml_path", default="assets/g1/scene_29dof_flat_hand.xml",
                   help="MuJoCo scene (deploy-root-relative) for VR-3-point FK.")
    p.add_argument("--task", default="walk up the stairs", help="Language instruction for GR00T.")
    p.add_argument("--control_dt", type=float, default=0.02, help="Channel publish period (s).")
    p.add_argument("--motion_fps", type=float, default=50.0,
                   help="FPS used to finite-difference reference joint velocity.")
    p.add_argument("--horizon", type=int, default=25, help="GR00T action horizon (H).")
    p.add_argument("--plan_latency", type=int, default=5,
                   help="Control steps of round-trip latency; replan at frame H-L, resume at L.")
    p.add_argument("--server_timeout", type=float, default=900.0,
                   help="Seconds to wait for vla:status == ready.")
    p.add_argument("--video_topic", action="append", default=[],
                   metavar="KEY=TOPIC",
                   help="Map a GR00T video key to a Redis camera topic (JPEG bytes). "
                        "Repeatable. Default: pelvis_left_view / pelvis_right_view.")
    args = p.parse_args()

    default_topics = [("pelvis_left_view", "camera:pelvis_left_view"),
                      ("pelvis_right_view", "camera:pelvis_right_view")]
    args.video_topics = _parse_video_topics(args.video_topic) if args.video_topic else default_topics

    VLAMotionBridge(args).run()


if __name__ == "__main__":
    main()
