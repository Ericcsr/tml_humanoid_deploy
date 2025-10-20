from argparse import ArgumentParser
import numpy as np
from scipy.spatial.transform import Rotation
from rl_policy import RLBMPolicy

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

def main(env, policy, config):
    
    if config["use_root_state"] and config.get("use_odom", False):
        kin_model = KinematicsModel(mocap_link_name="torso_link" if config["use_sim"] else "mid360_link", use_slam=config["use_slam"], visualize=False)

    env.set_robot_state(policy.get_q_init())
    env.maintain_state(policy.get_q_init())

    rate = Rate(1/config.get("control_dt", 0.02))  # 50 Hz

    robot_state = G1RobotState()


    env.release_robot()  # let the robot move
    init = False
    while True:
        robot_state.q, robot_state.dq, robot_state.imu_quat, robot_state.omega = env.get_robot_state()
        if config["use_root_state"]:
            if config.get("use_odom", False):
                robot_state.root_pos, robot_state.root_orn, robot_state.root_vel = kin_model.update_root_state(
                    q=robot_state.q,
                    dq=robot_state.dq,
                    imu_quat=robot_state.imu_quat,
                    omega=robot_state.omega,
                )
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
        robot_state.last_action = action.copy() # save last action

        scaled_action = action[ISAAC_TO_MUJOCO] * policy.action_scale + policy.default_value["q"][ISAAC_TO_MUJOCO]
        env.step_robot(scaled_action)
        rate.sleep()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--use_sim", action="store_true", help="Use simulation environment instead of real robot.")
    parser.add_argument("--use_odom", action="store_true", help="Use odometry for state estimation.")
    parser.add_argument("--use_slam", action="store_true", help="Use lidar for state estimation.")
    parser.add_argument("--net", type=str, required=False, help="Network interface for the robot controller.")
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = config["rl_policy"]

    # override config with command line args
    config["use_sim"] = args.use_sim
    config["use_odom"] = args.use_odom
    config["use_slam"] = args.use_slam

    if args.use_sim:
        from mujoco_env import MujocoRobot
        env = MujocoRobot(config["mujoco_xml_path"], config)
    else:
        from real_env import UnitreeRobot
        env = UnitreeRobot(args.net, config)

    lookahead_steps = config.get("lookahead_steps",1)
    lookahead_frame_skips = config.get("lookahead_frame_skips",1)
    policy = RLBMPolicy(config["onnx_model_path"], config["obs_names"], config["ref_motion_path"], 
                        lookahead_steps=lookahead_steps, lookahead_frame_skips=lookahead_frame_skips)

    main(env, policy, config)