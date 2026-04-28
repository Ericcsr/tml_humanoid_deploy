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


def main(env, policy, config, ticker_value=None, compute_metrics_flag=False):
    
    if config["use_root_state"] and config.get("use_odom", False):
        redis_client = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=0)

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
    motion_length = policy.motion_length
    init_offset = None  # (pos_offset, (robot_orn_0, ref_orn_0)) from frame 0 for first-frame alignment
    heading_align_rot = None  # rotation to align robot's initial heading with ref (removes heading-induced xy drift)

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
        
        action = policy.get_action(obs, start_ticker=env.get_start_ticker())

        # Sync ticker to ref motion visualizer (when use_sim)
        if ticker_value is not None:
            ticker_value.value = float(policy.ticker)

        # Compute metrics: robot state is from start of loop (after prev step); ticker was just incremented.
        # So robot = result of (ticker-1) steps, ref frame (ticker-1) is the one we used for that action.
        # Requires use_root_state for root_pos/root_orn.
        if compute_metrics_flag and config["use_root_state"]:
            mid = max(0, policy.ticker - 1) if policy.ticker > 0 else 0
            mid = min(mid, motion_length - 1)
            # Capture init offset at frame 0 for first-frame alignment (robot may start elsewhere than ref)
            if mid == 0 and init_offset is None:
                ref_anchor_pos_0 = policy.ref_anchor_poses[0]
                ref_anchor_orn_0 = policy.ref_anchor_orns[0]
                robot_rel_pos_0 = policy.init_root_heading_inv.apply(robot_state.root_pos - policy.init_root_pos)
                ref_rel_pos_0 = policy.init_root_heading_inv.apply(ref_anchor_pos_0 - policy.init_root_pos)
                robot_rel_orn_0 = (policy.init_root_heading_inv * Rotation.from_quat(robot_state.root_orn)).as_quat()
                ref_rel_orn_0 = (policy.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn_0)).as_quat()
                pos_offset = robot_rel_pos_0 - ref_rel_pos_0
                init_offset = (pos_offset, (robot_rel_orn_0, ref_rel_orn_0), ref_rel_pos_0)
                # Heading alignment: rotate robot's displacement so initial heading matches ref (removes xy drift)
                heading_diff = heading_zup(robot_rel_orn_0) - heading_zup(ref_rel_orn_0)
                heading_align_rot = Rotation.from_euler("z", -heading_diff)
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
            if policy.ticker >= motion_length:
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
                return robot_arr, ref_arr

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
    parser.add_argument("--slow_down", type=float, default=1.0, help="Slow down simulation and policy by x times (does not affect simulation_dt).")
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

    if args.slow_down != 1.0:
        print(f"[run_controller] Slow down: {args.slow_down}x (simulation_dt unchanged)", flush=True)

    # When terrain/object + sim: init robot at first frame xy and heading (for placement)
    terrain_urdf = config.get("terrain_urdf") or ""
    terrain_urdf = str(terrain_urdf).strip() if terrain_urdf else ""
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
    object_urdf = config.get("object_urdf") or ""
    object_urdf = str(object_urdf).strip() if object_urdf else ""
    object_motion = config.get("object_motion") or ""
    object_motion = str(object_motion).strip() if object_motion else ""
    has_object = bool(("object_urdf" in config and object_urdf) or ("object_motion" in config and object_motion))
    init_at_first_frame = bool(
        ("terrain_urdf" in config and terrain_urdf and args.use_sim)
        or (has_terrain_box and args.use_sim)
        or (has_object and args.use_sim)
    )
    if init_at_first_frame:
        from utils.params import ISAAC_TO_MUJOCO
        ref_motion = np.load(config["ref_motion_path"])
        config["sim_init_root_pos"] = ref_motion["body_pos_w"][0, 0].copy()
        config["sim_init_root_orn"] = ref_motion["body_quat_w"][0, 0][[1, 2, 3, 0]].copy()
        config["sim_init_joint_pos"] = ref_motion["joint_pos"][0, ISAAC_TO_MUJOCO].copy()
        print(f"[run_controller] Init at first frame: xy=({config['sim_init_root_pos'][0]:.2f}, {config['sim_init_root_pos'][1]:.2f}) z={config['sim_init_root_pos'][2]:.2f}", flush=True)

    if args.use_sim:
        from mujoco_env import MujocoRobot
        from ref_motion_visualizer import start_ref_visualizer_process

        ticker_value = Value("f", 0.0)
        env = MujocoRobot(config["mujoco_xml_path"], config, ticker_value=ticker_value)
        # Only start ref motion visualizer when --metric is enabled
        if args.metric:
            ref_vis_process = start_ref_visualizer_process(
                config["mujoco_xml_path"],
                config["ref_motion_path"],
                control_dt=config.get("control_dt", 0.02),
                ticker_value=ticker_value,
            )
        else:
            ref_vis_process = None
    else:
        from real_env import UnitreeRobot
        env = UnitreeRobot(args.net, config)
        ticker_value = None
        ref_vis_process = None

    lookahead_steps = config.get("lookahead_steps",1)
    lookahead_frame_skips = config.get("lookahead_frame_skips",1)
    if args.vr:
        policy = RL3ptPolicy(config["onnx_model_path"], config["obs_names"], config["ref_motion_path"], 
                            lookahead_steps=lookahead_steps, lookahead_frame_skips=lookahead_frame_skips,
                            init_at_first_frame=init_at_first_frame)
    elif config.get("use_chip", False):
        if config.get("only_3pt", False):
            policy = RLGlobalCHIPPolicy(config["onnx_model_path"], config["obs_names"], config["ref_motion_path"], 
                            lookahead_steps=lookahead_steps, lookahead_frame_skips=lookahead_frame_skips,
                            hist_names=config.get("history_names", []), hist_length=config.get("history_length", 1),
                            init_at_first_frame=init_at_first_frame)
        else:
            policy = RLCHIPPolicy(config["onnx_model_path"], config["obs_names"], config["ref_motion_path"], 
                                lookahead_steps=lookahead_steps, lookahead_frame_skips=lookahead_frame_skips,
                                hist_names=config.get("history_names", []), hist_length=config.get("history_length", 1),
                                init_at_first_frame=init_at_first_frame)                    
    elif config.get("use_streaming_motion", False):
        policy = RLStreamingContactPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            hist_names=config.get("history_names", []),
            hist_length=config.get("history_length", 1),
            redis_ip=config.get("streaming_redis_ip", REDIS_IP),
            redis_port=config.get("streaming_redis_port", REDIS_PORT),
            redis_channels=config.get("streaming_channels", None),
            default_contact_label=config.get("default_contact_label", None),
            use_8way_contact=config.get("use_8way_contact", False),
        )
    elif config.get("use_contact", False):
        policy = RLContactPolicy(
            config["onnx_model_path"],
            config["obs_names"],
            config["ref_motion_path"],
            config["contact_labels_path"],
            lookahead_steps=lookahead_steps,
            lookahead_frame_skips=lookahead_frame_skips,
            hist_names=config.get("history_names", []),
            hist_length=config.get("history_length", 1),
            init_at_first_frame=init_at_first_frame,
            zero_foot_contact_on_load=config.get("zero_foot_contact_on_load", False),
            use_8way_contact=config.get("use_8way_contact", False),
        )
    else:
        policy = RLBMPolicy(config["onnx_model_path"], config["obs_names"], config["ref_motion_path"], 
                            lookahead_steps=lookahead_steps, lookahead_frame_skips=lookahead_frame_skips,
                            init_at_first_frame=init_at_first_frame)

    try:
        result = main(env, policy, config, ticker_value=ticker_value, compute_metrics_flag=args.metric)
        if result is not None and args.metric:
            robot_traj, ref_traj = result
            from trajectory_visualizer import plot_root_trajectories
            plot_root_trajectories(robot_traj, ref_traj)
    finally:
        if ref_vis_process is not None:
            ref_vis_process.terminate()
            ref_vis_process.join()