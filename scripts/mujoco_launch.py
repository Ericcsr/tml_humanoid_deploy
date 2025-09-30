#!/usr/bin/env python3
"""
MuJoCo Simulation Controller
"""

from argparse import ArgumentParser
import yaml
from tml_humanoid_deploy.agents.rl_agent import RLBMAgent
from tml_humanoid_deploy.envs.mujoco_env import MujocoRobot
from scripts.run_controller import run_controller

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--use_odom", action="store_true", help="Use odometry for state estimation.")
    parser.add_argument("--net", type=str, required=False, help="Network interface for the robot controller.")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        rl_config = config["rl_policy"]
        standby_config = config.get("standby_policy", {})
    
    rl_config["use_sim"] = True
    rl_config["use_odom"] = args.use_odom

    env = MujocoRobot(rl_config["mujoco_xml_path"], rl_config)
    policy = RLBMAgent(rl_config["onnx_model_path"], rl_config["obs_names"], rl_config["ref_motion_path"])
    
    print("Starting MuJoCo Simulation...")
    run_controller(env, policy, rl_config, standby_config)

if __name__ == "__main__":
    main()
