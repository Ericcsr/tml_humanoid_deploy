from argparse import ArgumentParser
import numpy as np
import redis
import time
import pickle
from multiprocessing import Value
from utils.redis_utils import REDIS_IP, REDIS_PORT
from scipy.spatial.transform import Rotation
from rl_policy import (
    RLBMPolicy,
    RL3ptPolicy,
    RLCHIPPolicy,
    RLContactPolicy,
    RLGlobalCHIPPolicy,
    RLStreamingContactPolicy,
    clamp_ref_motion_start_index,
    load_ref_motion_with_optional_slowdown,
)

from utils.params import DEFAULT_POSE, ACTION_SCALE, ISAAC_TO_MUJOCO
from utils.robot_utils import Rate
from utils.robot_states import G1RobotState

from utils.robot_model import KinematicsModel

def rotatepoint(q, v):
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[3:4]
    q_xyz = q[:3]
    t = 2 * np.cross(q_xyz, v)
    return v + q_r * t + np.cross(q_xyz, t)

def heading_zup(quat):
    ref_dir = np.zeros_like(quat[:3])
    ref_dir[0] = 1
    ref_dir = rotatepoint(quat, ref_dir)
    return np.arctan2(ref_dir[1], ref_dir[0])

def compute_metrics(robot_state, policy, mid, init_offset=None):
    """Compute per-frame errors between robot and reference.
    Both use policy's init_root_heading_inv to transform to init-relative frame.
    When init_offset is provided (from frame 0), we subtract it to measure tracking error
    rather than absolute offset (robot may start at different position than reference).
    For trajectory: use per-frame ref-frame alignment to remove heading-induced xy drift.
    """
    ref_q = policy.ref_q_pos[mid][ISAAC_TO_MUJOCO]
    ref_anchor_pos = policy.ref_anchor_poses[mid]
    ref_anchor_orn = policy.ref_anchor_orns[mid]

    # Same transform as policy: both to init-relative frame (origin at init_root_pos, yaw-aligned)
    robot_rel_pos = policy.init_root_heading_inv.apply(robot_state.root_pos - policy.init_root_pos)
    ref_rel_pos = policy.init_root_heading_inv.apply(ref_anchor_pos - policy.init_root_pos)
    robot_rel_orn = (policy.init_root_heading_inv * Rotation.from_quat(robot_state.root_orn)).as_quat()
    ref_rel_orn = (policy.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn)).as_quat()

    # First-frame alignment: subtract initial pos offset from robot only (align robot to ref at frame 0)
    if init_offset is not None:
        pos_offset, (robot_orn_0, ref_orn_0) = init_offset[0], init_offset[1]
        robot_rel_pos = robot_rel_pos - pos_offset
        # Orientation error = angle between robot's pose change and ref's pose change (from frame 0)
        robot_delta = Rotation.from_quat(robot_orn_0).inv() * Rotation.from_quat(robot_rel_orn)
        ref_delta = Rotation.from_quat(ref_orn_0).inv() * Rotation.from_quat(ref_rel_orn)
        root_orn_err = (robot_delta * ref_delta.inv()).magnitude()
    else:
        root_orn_err = (Rotation.from_quat(robot_rel_orn) * Rotation.from_quat(ref_rel_orn).inv()).magnitude()

    joint_err = np.mean(np.abs(robot_state.q - ref_q))
    root_pos_err = np.linalg.norm(robot_rel_pos - ref_rel_pos)

    return joint_err, root_pos_err, root_orn_err, robot_rel_pos.copy(), ref_rel_pos.copy()


# Reference-motion body names for the left / right hand end effectors. Resolved to
# column indices from the motion's ``body_names`` at runtime (fallback 28/29).
REF_LEFT_HAND_BODY = "left_wrist_yaw_link"
REF_RIGHT_HAND_BODY = "right_wrist_yaw_link"


def _ref_hand_indices(policy):
    """(left, right) hand body column indices in policy.ref_motion['body_pos_w']."""
    ref_motion = policy.ref_motion
    if "body_names" in getattr(ref_motion, "files", []) or (
        isinstance(ref_motion, dict) and "body_names" in ref_motion
    ):
        names = [str(n) for n in np.asarray(ref_motion["body_names"])]
        return names.index(REF_LEFT_HAND_BODY), names.index(REF_RIGHT_HAND_BODY)
    return 28, 29


def _to_init_relative(policy, world_pos):
    """World point -> init-relative frame (origin at init_root_pos, yaw-aligned)."""
    return policy.init_root_heading_inv.apply(world_pos - policy.init_root_pos)


def _align_robot_point(policy, world_pos, init_offset, heading_align_rot):
    """Robot world point -> init-relative frame with the same first-frame alignment
    (position offset + heading rotation) applied to the root trajectory, so robot
    and reference are directly comparable."""
    rel = _to_init_relative(policy, world_pos)
    pos_offset, _, ref_rel_pos_0 = init_offset
    rel = rel - pos_offset
    if heading_align_rot is not None:
        rel = ref_rel_pos_0 + heading_align_rot.apply(rel - ref_rel_pos_0)
    return rel


def main(env, policy, config, ticker_value=None, compute_metrics_flag=False,
         eef_error_flag=False, eef_fk=None, init_at_ref=False, hold_frame0_sec=0.0):

    if config["use_root_state"] and config.get("use_odom", False):
        redis_client = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=0)

    if init_at_ref:
        # Robot is already spawned exactly at the reference start frame with the band
        # disabled; just command the reference joints (no ramp, no Press-Enter) so the
        # policy starts from ~zero frame-0 error.
        env.begin_at_reference(policy.get_q_init())
    else:
        env.set_robot_state(policy.get_q_init())
        env.maintain_state(policy.get_q_init())

    control_dt = config.get("control_dt", 0.02)
    slow_down = config.get("slow_down", 1.0)
    rate = Rate(1 / (control_dt * slow_down))

    robot_state = G1RobotState()

    if config["use_root_state"] and config.get("use_odom", False):
        robot_state.q, robot_state.dq, robot_state.imu_quat, robot_state.omega = env.get_robot_state()
        redis_client.set("proprio_data", pickle.dumps(np.hstack((
                        robot_state.q,
                        robot_state.dq,
                        robot_state.omega,
                        robot_state.imu_quat,
        ))))

    env.release_robot()  # let the robot move
    init = False
    if config["use_root_state"] and config.get("use_odom", False):
        while redis_client.get("root_data") is None:
            rate.sleep()

    # Accumulators for metrics (when --metric enabled)
    joint_errors, root_pos_errors, root_orn_errors = [], [], []
    robot_traj, ref_traj = [], []  # root trajectories for visualization
    # End-effector (hand) world-position accumulators (when --eef_error enabled)
    robot_lhand, ref_lhand, robot_rhand, ref_rhand = [], [], [], []
    motion_length = policy.motion_length
    init_offset = None  # alignment from ref_motion_start_index frame when --metric/--eef_error
    heading_align_rot = None  # rotation to align robot's initial heading with ref (removes heading-induced xy drift)
    track_flag = compute_metrics_flag or eef_error_flag
    if eef_error_flag:
        ref_lhand_idx, ref_rhand_idx = _ref_hand_indices(policy)

    # Hold at frame 0: run the policy (so it actively balances) but freeze the reference
    # ticker for the first hold_frame0_sec seconds, letting the robot settle before the
    # motion starts advancing.
    hold_frame0_steps = int(round(hold_frame0_sec / control_dt)) if hold_frame0_sec > 0 else 0
    step_count = 0
    if hold_frame0_steps > 0:
        print(f"[run_controller] Holding at frame 0 for {hold_frame0_sec:g}s "
              f"({hold_frame0_steps} steps) before advancing the reference.", flush=True)

    while True:
        
        robot_state.q, robot_state.dq, robot_state.imu_quat, robot_state.omega = env.get_robot_state()
        if config["use_root_state"]:
            if config.get("use_odom", False):
                redis_client.set("proprio_data", pickle.dumps(np.hstack((
                    robot_state.q,
                    robot_state.dq,
                    robot_state.omega,
                    robot_state.imu_quat,
                ))))
                root_data = pickle.loads(redis_client.get("root_data"))
                robot_state.root_pos = root_data[:3]
                robot_state.root_orn = root_data[3:7]
                robot_state.root_vel = root_data[7:10]
            else:
                robot_state.root_pos, robot_state.root_orn, robot_state.root_vel = env.get_root_state()
                robot_state.anchor_pos, robot_state.anchor_orn = env.get_anchor_state()
        else:
            if not init:
                init_orn = robot_state.imu_quat
                init_heading = heading_zup(init_orn)
                init_heading_rot = Rotation.from_euler("z", init_heading)
                init = True
            robot_state.root_orn = (init_heading_rot.inv() * Rotation.from_quat(robot_state.imu_quat)).as_quat()
        control_signals = policy.prepare_control_signals(robot_state)
        obs = policy.prepare_obs(robot_state, control_signals) # Should be reference motion.
        
        # Freeze the ticker during the frame-0 hold window, then advance normally.
        advance_ticker = env.get_start_ticker() and (step_count >= hold_frame0_steps)
        action = policy.get_action(obs, start_ticker=advance_ticker)
        step_count += 1

        # Sync ticker to ref motion visualizer (when use_sim)
        if ticker_value is not None:
            ticker_value.value = float(policy.ticker)

        # Compute metrics: robot state is from start of loop (after prev step); ticker was just incremented.
        # So robot = result of (ticker-1) steps, ref frame (ticker-1) is the one we used for that action.
        # Requires use_root_state for root_pos/root_orn.
        if track_flag and config["use_root_state"]:
            si = getattr(policy, "ref_motion_start_index", 0)
            if policy.ticker > si:
                mid = policy.ticker - 1
            else:
                mid = si
            mid = min(mid, motion_length - 1)
            # Capture init offset at start frame for first-frame alignment (robot may start elsewhere than ref)
            if mid == si and init_offset is None:
                ref_anchor_pos_0 = policy.ref_anchor_poses[si]
                ref_anchor_orn_0 = policy.ref_anchor_orns[si]
                robot_rel_pos_0 = policy.init_root_heading_inv.apply(robot_state.root_pos - policy.init_root_pos)
                ref_rel_pos_0 = policy.init_root_heading_inv.apply(ref_anchor_pos_0 - policy.init_root_pos)
                robot_rel_orn_0 = (policy.init_root_heading_inv * Rotation.from_quat(robot_state.root_orn)).as_quat()
                ref_rel_orn_0 = (policy.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn_0)).as_quat()
                pos_offset = robot_rel_pos_0 - ref_rel_pos_0
                init_offset = (pos_offset, (robot_rel_orn_0, ref_rel_orn_0), ref_rel_pos_0)
                # Heading alignment: rotate robot's displacement so initial heading matches ref (removes xy drift)
                heading_diff = heading_zup(robot_rel_orn_0) - heading_zup(ref_rel_orn_0)
                heading_align_rot = Rotation.from_euler("z", -heading_diff)

            if compute_metrics_flag:
                je, rpe, roe, robot_pos, ref_pos = compute_metrics(robot_state, policy, mid, init_offset)
                joint_errors.append(je)
                root_pos_errors.append(rpe)
                root_orn_errors.append(roe)
                # Apply heading alignment to robot trajectory (robot_pos already has pos_offset applied)
                if heading_align_rot is not None:
                    _, _, ref_rel_pos_0 = init_offset
                    robot_disp = robot_pos - ref_rel_pos_0
                    robot_pos_aligned = ref_rel_pos_0 + heading_align_rot.apply(robot_disp)
                    robot_traj.append(robot_pos_aligned)
                else:
                    robot_traj.append(robot_pos)
                ref_traj.append(ref_pos)

            # End-effector tracking: robot hand poses via FK vs reference-motion hand poses,
            # both mapped to the init-relative frame with the same first-frame alignment.
            if eef_error_flag and eef_fk is not None and init_offset is not None:
                l_pos_w, _, r_pos_w, _ = eef_fk.hand_world_poses(
                    robot_state.q, robot_state.root_pos, robot_state.root_orn
                )
                robot_lhand.append(_align_robot_point(policy, l_pos_w, init_offset, heading_align_rot))
                robot_rhand.append(_align_robot_point(policy, r_pos_w, init_offset, heading_align_rot))
                ref_lhand.append(_to_init_relative(policy, policy.ref_motion["body_pos_w"][mid, ref_lhand_idx]))
                ref_rhand.append(_to_init_relative(policy, policy.ref_motion["body_pos_w"][mid, ref_rhand_idx]))

            if policy.ticker >= motion_length:
                result = {}
                if compute_metrics_flag:
                    mean_je = np.mean(joint_errors)
                    # Use same trajectory data as plot (heading-aligned robot vs ref) for printed metric
                    robot_arr = np.array(robot_traj)
                    ref_arr = np.array(ref_traj)
                    mean_rpe = np.mean(np.linalg.norm(robot_arr - ref_arr, axis=1))
                    mean_roe = np.mean(root_orn_errors)
                    print("\n=== Motion Rollout Metrics ===")
                    print(f"  Mean joint error (rad):     {mean_je:.6f}")
                    print(f"  Mean root position error (m): {mean_rpe:.6f}")
                    print(f"  Mean root orientation error (rad): {mean_roe:.6f}")
                    print("==============================\n")
                    result["robot_traj"] = robot_arr
                    result["ref_traj"] = ref_arr
                if eef_error_flag:
                    rl, rfl = np.array(robot_lhand), np.array(ref_lhand)
                    rr, rfr = np.array(robot_rhand), np.array(ref_rhand)
                    mean_le = np.mean(np.linalg.norm(rl - rfl, axis=1)) if len(rl) else float("nan")
                    mean_re = np.mean(np.linalg.norm(rr - rfr, axis=1)) if len(rr) else float("nan")
                    print("=== End-Effector Tracking Metrics ===")
                    print(f"  Mean left hand position error (m):  {mean_le:.6f}")
                    print(f"  Mean right hand position error (m): {mean_re:.6f}")
                    print("=====================================\n")
                    result["robot_lhand"], result["ref_lhand"] = rl, rfl
                    result["robot_rhand"], result["ref_rhand"] = rr, rfr
                return result

        robot_state.last_action = action.copy() # save last action

        scaled_action = action[ISAAC_TO_MUJOCO] * policy.action_scale + policy.default_value["q"][ISAAC_TO_MUJOCO]
        env.step_robot(scaled_action)
        rate.sleep()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--use_sim", action="store_true", help="Use simulation environment instead of real robot.")
    parser.add_argument("--use_odom", action="store_true", help="Use odometry for state estimation.")
    parser.add_argument("--vr", action="store_true", default=False, help="Use 3-point VR controller.")
    parser.add_argument("--net", type=str, required=False, help="Network interface for the robot controller.")
    parser.add_argument("--metric", action="store_true", help="Compute mean joint/root errors vs reference during rollout, then exit.")
    parser.add_argument("--vis_ref", action="store_true", help="Show the reference motion in a MuJoCo viewer alongside the robot (sim only).")
    parser.add_argument("--init_at_ref", action="store_true", help="Spawn the robot exactly at the reference start frame (root + joints), no elastic band and no ramp/pause, then start the policy immediately (sim only). Guarantees ~zero frame-0 tracking error.")
    parser.add_argument("--hold_frame0_sec", type=float, default=0.0, help="Hold at the reference frame 0 for this many seconds (policy runs to balance, reference ticker frozen) before the motion starts advancing, so the robot settles and does not fall at frame 0.")
    parser.add_argument("--eef_error", action="store_true", help="Record left/right end-effector (hand) GT vs real poses and save an error plot PNG, then exit.")
    parser.add_argument("--eef_error_png", type=str, default="eef_error.png", help="Output path for the --eef_error plot PNG.")
    parser.add_argument("--slow_down", type=float, default=1.0, help="Slow down simulation and policy by x times (does not affect simulation_dt).")
    parser.add_argument("--session-id", type=str, default=None, help="Per-session suffix for redis keys + shared-memory blocks. Used by spawn_server.py for per-user isolation; leave unset for the default single-sim path.")
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = config["rl_policy"]

    # override config with command line args
    config["use_sim"] = args.use_sim
    config["use_odom"] = args.use_odom
    config["slow_down"] = args.slow_down

    if args.metric and not config.get("use_root_state", False):
        raise ValueError("--metric requires use_root_state: True in config (for root position/orientation).")
    if args.eef_error and not config.get("use_root_state", False):
        raise ValueError("--eef_error requires use_root_state: True in config (for root position/orientation).")

    if args.slow_down != 1.0:
        print(f"[run_controller] Slow down: {args.slow_down}x (simulation_dt unchanged)", flush=True)

    # When terrain/object + sim: init robot at first frame xy and heading (for placement)
    terrain_urdf = config.get("terrain_urdf") or ""
    terrain_urdf = str(terrain_urdf).strip() if terrain_urdf else ""
    terrain_mujoco_xml = config.get("terrain_mujoco_xml") or ""
    terrain_mujoco_xml = str(terrain_mujoco_xml).strip() if terrain_mujoco_xml else ""
    terrain_box_pos = config.get("terrain_box_pos")
    terrain_box_size = config.get("terrain_box_size")
    has_terrain_box_legacy = terrain_box_pos is not None and terrain_box_size is not None
    tbc = config.get("terrain_boxes")
    if isinstance(tbc, dict):
        has_terrain_box_multi = len(tbc) > 0
    elif isinstance(tbc, list):
        has_terrain_box_multi = len(tbc) > 0
    else:
        has_terrain_box_multi = False
    has_terrain_box = has_terrain_box_legacy or has_terrain_box_multi
    twc = config.get("terrain_wedges")
    if isinstance(twc, (dict, list)):
        has_terrain_wedge = len(twc) > 0
    else:
        has_terrain_wedge = False
    object_urdf = config.get("object_urdf") or ""
    object_urdf = str(object_urdf).strip() if object_urdf else ""
    object_motion = config.get("object_motion") or ""
    object_motion = str(object_motion).strip() if object_motion else ""
    has_object = bool(("object_urdf" in config and object_urdf) or ("object_motion" in config and object_motion))
    init_at_first_frame = bool(
        ("terrain_urdf" in config and terrain_urdf and args.use_sim)
        or ("terrain_mujoco_xml" in config and terrain_mujoco_xml and args.use_sim)
        or (has_terrain_box and args.use_sim)
        or (has_terrain_wedge and args.use_sim)
        or (has_object and args.use_sim)
        # --init_at_ref forces the world-frame (first-frame) spawn even on flat/freespace,
        # so the robot starts exactly on the reference and the policy runs un-recentered.
        or (args.init_at_ref and args.use_sim)
    )
    slow_motion_end_frame = config.get("slow_motion_end_frame", None)
    slow_down_times = config.get("slow_down_times", None)
    _ref_for_start_index = None
    if config.get("ref_motion_path"):
        _ref_for_start_index = load_ref_motion_with_optional_slowdown(
            config["ref_motion_path"],
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
        config["ref_motion_start_index"] = clamp_ref_motion_start_index(
            int(config.get("ref_motion_start_index", 0)),
            int(_ref_for_start_index["joint_pos"].shape[0]),
        )
    else:
        config["ref_motion_start_index"] = 0
    if init_at_first_frame:
        from utils.params import ISAAC_TO_MUJOCO
        ref_motion = _ref_for_start_index
        if ref_motion is None:
            ref_motion = np.load(config["ref_motion_path"])
        si = config["ref_motion_start_index"]
        config["sim_init_root_pos"] = ref_motion["body_pos_w"][si, 0].copy()
        config["sim_init_root_orn"] = ref_motion["body_quat_w"][si, 0][[1, 2, 3, 0]].copy()
        config["sim_init_joint_pos"] = ref_motion["joint_pos"][si, ISAAC_TO_MUJOCO].copy()
        print(
            f"[run_controller] Init at ref frame {si}: xy=({config['sim_init_root_pos'][0]:.2f}, "
            f"{config['sim_init_root_pos'][1]:.2f}) z={config['sim_init_root_pos'][2]:.2f}",
            flush=True,
        )

    if args.use_sim:
        from mujoco_env import MujocoRobot

        # Show the reference motion as a transparent, collision-free ghost in the SAME
        # viewer when explicitly requested (--vis_ref) or a comparison flag is enabled.
        if (args.metric or args.vis_ref or args.eef_error) and config.get("ref_motion_path"):
            config["show_reference_ghost"] = True
            # Match the policy's frame choice so the ghost lines up with the robot.
            config["init_at_first_frame_ghost"] = init_at_first_frame

        # --init_at_ref: tell the sim child to spawn exactly on the reference start frame,
        # disable the elastic band, and hold the reference joints from step 0 (no ramp).
        if args.init_at_ref:
            if config.get("sim_init_joint_pos") is None:
                raise ValueError("--init_at_ref requires a ref_motion_path in the config.")
            config["init_at_ref"] = True

        ticker_value = Value("f", float(config["ref_motion_start_index"]))
        env = MujocoRobot(config["mujoco_xml_path"], config, ticker_value=ticker_value, session_id=args.session_id)
    else:
        from real_env import UnitreeRobot
        env = UnitreeRobot(args.net, config)
        ticker_value = None

    lookahead_steps = config.get("lookahead_steps",1)
    lookahead_frame_skips = config.get("lookahead_frame_skips",1)
    if args.vr:
        policy = RL3ptPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            config["ref_motion_path"],
            use_sim=args.use_sim,
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            init_at_first_frame=init_at_first_frame,
            ref_motion_start_index=config["ref_motion_start_index"],
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
    elif config.get("use_chip", False):
        if config.get("only_3pt", False):
            policy = RLGlobalCHIPPolicy(
                config["onnx_model_path"],
                config["obs_names"],
                config["ref_motion_path"],
                use_sim=args.use_sim,
                lookahead_steps=lookahead_steps,
                lookahead_frame_skips=lookahead_frame_skips,
                hist_names=config.get("history_names", []),
                hist_length=config.get("history_length", 1),
                init_at_first_frame=init_at_first_frame,
                ref_motion_start_index=config["ref_motion_start_index"],
                slow_motion_end_frame=slow_motion_end_frame,
                slow_down_times=slow_down_times,
            )
        else:
            policy = RLCHIPPolicy(
                config["onnx_model_path"],
                config["obs_names"],
                config["ref_motion_path"],
                use_sim=args.use_sim,
                lookahead_steps=lookahead_steps,
                lookahead_frame_skips=lookahead_frame_skips,
                hist_names=config.get("history_names", []),
                hist_length=config.get("history_length", 1),
                init_at_first_frame=init_at_first_frame,
                ref_motion_start_index=config["ref_motion_start_index"],
                slow_motion_end_frame=slow_motion_end_frame,
                slow_down_times=slow_down_times,
            )                    
    elif config.get("use_streaming_motion", False):
        # When --session-id is set, suffix every streaming channel name with
        # ":<sid>" so this controller only sees its own session's motion graph
        # output. Without this, two concurrent sessions' motion graphs would
        # publish to the same channel and the policies would receive a mix.
        _streaming_channels = config.get("streaming_channels", None)
        if args.session_id:
            _base_channels = _streaming_channels or {
                "lower_cmd": "lower_cmd",
                "vr_3point_pos_l": "vr_3point_pos_l",
                "vr_3point_orn_l": "vr_3point_orn_l",
                "contact_mask": "contact_mask",
                "motion_anchor_pos_w": "motion_anchor_pos_w",
                "motion_anchor_orn_w": "motion_anchor_orn_w",
            }
            _streaming_channels = {k: f"{v}:{args.session_id}" for k, v in _base_channels.items()}
        policy = RLStreamingContactPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            use_sim=args.use_sim,
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            hist_names=config.get("history_names", []),
            hist_length=config.get("history_length", 1),
            redis_ip=config.get("streaming_redis_ip", REDIS_IP),
            redis_port=config.get("streaming_redis_port", REDIS_PORT),
            redis_channels=_streaming_channels,
            default_contact_label=config.get("default_contact_label", None),
            use_8way_contact=config.get("use_8way_contact", False),
            use_10way_contact=config.get("use_10way_contact", False),
            use_5dim_contact_from_4dim=config.get("use_5dim_contact_from_4dim", False),
            wrist_contact_as_env=config.get("wrist_contact_as_env", False),
            disable_hand_contact_labels=config.get("disable_hand_contact_labels", False),
            ref_motion_start_index=config["ref_motion_start_index"],
        )
    elif config.get("use_contact", False):
        policy = RLContactPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            config["ref_motion_path"],
            config["contact_labels_path"],
            use_sim=args.use_sim,
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            hist_names=config.get("history_names", []),
            hist_length=config.get("history_length", 1),
            init_at_first_frame=init_at_first_frame,
            zero_foot_contact_on_load=config.get("zero_foot_contact_on_load", False),
            use_8way_contact=config.get("use_8way_contact", False),
            use_10way_contact=config.get("use_10way_contact", False),
            use_5dim_contact_from_4dim=config.get("use_5dim_contact_from_4dim", False),
            wrist_contact_as_env=config.get("wrist_contact_as_env", False),
            disable_hand_contact_labels=config.get("disable_hand_contact_labels", False),
            ref_motion_start_index=config["ref_motion_start_index"],
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
    else:
        policy = RLBMPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            config["ref_motion_path"],
            use_sim=args.use_sim,
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            init_at_first_frame=init_at_first_frame,
            ref_motion_start_index=config["ref_motion_start_index"],
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )

    # Forward-kinematics helper for the robot's achieved hand poses (--eef_error).
    eef_fk = None
    if args.eef_error:
        from eef_tracker import HandForwardKinematics
        eef_fk = HandForwardKinematics(config["mujoco_xml_path"])

    try:
        result = main(
            env, policy, config,
            ticker_value=ticker_value,
            compute_metrics_flag=args.metric,
            eef_error_flag=args.eef_error,
            eef_fk=eef_fk,
            init_at_ref=args.init_at_ref,
            hold_frame0_sec=args.hold_frame0_sec,
        )
        if result:
            if args.metric and "robot_traj" in result:
                from trajectory_visualizer import plot_root_trajectories
                plot_root_trajectories(result["robot_traj"], result["ref_traj"])
            if args.eef_error and "robot_lhand" in result:
                from trajectory_visualizer import plot_eef_errors
                plot_eef_errors(
                    result["robot_lhand"], result["ref_lhand"],
                    result["robot_rhand"], result["ref_rhand"],
                    output_png=args.eef_error_png,
                )
    finally:
        if args.use_sim and hasattr(env, "close"):
            env.close()