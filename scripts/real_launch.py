#!/usr/bin/env python3
"""
Real Robot Controller
DANGER: This will control the actual physical robot!
Only run this when you're ready to move the real robot.
"""

from argparse import ArgumentParser
import yaml
from tml_humanoid_deploy.agents.rl_agent import RLBMAgent
from tml_humanoid_deploy.envs.real_env import UnitreeRobot
from scripts.run_controller import run_controller

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--use_odom", action="store_true", help="Use odometry for state estimation.")
    parser.add_argument("--net", type=str, required=True, help="Network interface for the robot controller.")
    args = parser.parse_args()
    
    # Safety confirmation
    print("⚠️  WARNING: This will control the REAL ROBOT!")
    input("Press ENTER to confirm you want to control the real robot (Ctrl+C to abort): ")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        rl_config = config["rl_policy"]
        standby_config = config.get("standby_policy", {})

    # Force real robot mode
    rl_config["use_sim"] = False
    rl_config["use_odom"] = args.use_odom
    
    env = UnitreeRobot(args.net, rl_config)
    policy = RLBMAgent(rl_config["onnx_model_path"], rl_config["obs_names"], rl_config["ref_motion_path"])
    
    print("🤖 Starting Real Robot Controller...")
    run_controller(env, policy, rl_config, standby_config)

if __name__ == "__main__":
    main()
