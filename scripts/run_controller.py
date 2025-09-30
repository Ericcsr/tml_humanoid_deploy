from argparse import ArgumentParser
import numpy as np
import yaml
from typing import Dict, Any
from tml_humanoid_deploy.agents.base_agent import BaseAgent
from tml_humanoid_deploy.envs.base_env import BaseRobotEnv

from tml_humanoid_deploy.utils.params import DEFAULT_POSE, ACTION_SCALE, ISAAC_TO_MUJOCO
from tml_humanoid_deploy.utils.robot_utils import Rate
from tml_humanoid_deploy.utils.robot_states import G1RobotState

from tml_humanoid_deploy.utils.robot_model import KinematicsModel

def run_controller(
    env: BaseRobotEnv, 
    policy: BaseAgent, 
    config: Dict[str, Any], 
    standby_config: Dict[str, Any]
) -> None:
    
    if config["use_root_state"] and config.get("use_odom", False):
        kin_model = KinematicsModel(mocap_link_name="mid360_link" if config["use_sim"] else "head_link", use_slam=False, visualize=True)
    
    # Get standby pose and gains
    standby_pose = np.array(standby_config.get("default_pose", policy.get_q_init()), dtype=np.float32)
    standby_kp = np.array(standby_config["joint_stiffness"], dtype=np.float32) if "joint_stiffness" in standby_config else None
    standby_kd = np.array(standby_config["joint_damping"], dtype=np.float32) if "joint_damping" in standby_config else None
    
    # Set robot to standby state with standby gains
    env.set_robot_state(standby_pose, kp=standby_kp, kd=standby_kd)
    env.maintain_state(standby_pose, kp=standby_kp, kd=standby_kd)
    
    rate = Rate(1/config.get("control_dt", 0.02))  # 50 Hz
    robot_state = G1RobotState()
    
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
                robot_state.root_orn = robot_state.imu_quat # wxyz
            else:
                robot_state.root_pos, robot_state.root_orn, robot_state.root_vel = env.get_root_state()
                robot_state.anchor_pos, robot_state.anchor_orn = env.get_anchor_state()
        
        control_signals = policy.prepare_control_signals(robot_state)
        obs = policy.prepare_obs(robot_state, control_signals) # Should be reference motion.
        
        action = policy.get_action(obs)
        robot_state.last_action = action.copy() # save last action

        scaled_action = action[ISAAC_TO_MUJOCO] * policy.action_scale + policy.default_value["q"][ISAAC_TO_MUJOCO]
        env.step_robot(scaled_action)
        rate.sleep()