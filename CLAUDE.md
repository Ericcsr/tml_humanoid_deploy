# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based humanoid robot control deployment for the Unitree G1. Executes ONNX policies trained in Isaac Gym, supporting sim-to-sim (MuJoCo) and sim-to-real transfer with state estimation.

## Environment Setup

```bash
conda create -n bm python=3.10.18
conda activate bm
pip install mujoco joblib onnxruntime pybullet scipy torch PyYAML opencv-python redis
sudo apt install redis-server
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e . --no-deps
```

## Running

```bash
# Sim2Sim with ground truth state
python run_controller.py --use_sim --config exported_policies/sirui_test/experiment.yaml

# Sim2Real (sim) with foot odometry — requires two terminals
# Terminal 1:
python run_controller.py --use_sim --config exported_policies/sirui_test/experiment.yaml --use_odom
# Terminal 2:
python run_state_estimation.py --use_sim --visualize

# With SLAM fusion (requires external lidar SLAM: github.com/Ericcsr/G1_localization)
python run_controller.py --use_sim --config exported_policies/sirui_test/experiment.yaml --use_odom --use_slam
python run_state_estimation.py --use_sim --visualize --use_slam

# Motion replay
python replay_motion.py --motion <path_to_motion.npz>
```

Key flags: `--use_sim` (MuJoCo instead of real robot), `--use_odom` (foot odometry state estimation), `--use_slam` (SLAM fusion), `--visualize` (state estimation viewer), `--use_acc` (acceleration-based odometry), `--metric` (tracking error plots).

## Architecture

**Control loop** (`run_controller.py`): Assembles robot state, runs RL policy inference at 50 Hz (control_dt=0.02s), sends torque-limited PD commands. Physics runs at 500 Hz (simulation_dt=0.002s).

**Policy classes** (`rl_policy.py`): ONNX inference with observation queuing and lookahead.
- `RLBMPolicy` — body motion with reference motion tracking (main policy)
- `RL3ptPolicy` — 3-point VR-based control
- `RLCHIPPolicy` — compliance controller (keyboard: g=stiff, h=compliant, j=partial)

**Environments**: `mujoco_env.py` (simulation) and `real_env.py` (hardware via Unitree DDS SDK) share a common interface.

**State estimation** (`run_state_estimation.py`, `utils/state_estimation.py`): Foot odometry + optional SLAM fusion. Publishes root pose/velocity over Redis (localhost:6379). Kalman filtering with Numba JIT-compiled kernels.

**IPC**: Redis connects state estimation and control processes. State estimation publishes `root_data`; controller subscribes.

## Coordinate Frames and Joint Indexing

- Isaac Gym and MuJoCo use different joint orderings. `utils/params.py` defines `ISAAC_TO_MUJOCO` and `MUJOCO_TO_ISAAC` mappings (29 DOF).
- `ACTION_SCALE` in `params.py` applies per-joint scaling to policy outputs.
- Anchor/root link is **pelvis** (not torso).
- Reference motions use init-relative frame alignment in `RLBMPolicy`.

## Experiment Configuration

Each experiment lives in `exported_policies/<name>/` with:
- `experiment.yaml` — obs names, control/sim dt, joint stiffness (Kp), damping (Kd), torque limits
- `policy.onnx` — trained policy from Isaac Gym
- `ref_motion.npz` — reference trajectories (keys: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`)

## Notes

- Sim2Real code is untested on hardware.
- Robot model: `assets/g1/g1_29dof.xml` (MuJoCo), `g1_29dof_kin_extended.urdf` (PyBullet kinematics).
- `utils/robot_model.py` provides FK/IK via PyBullet and ElasticBand virtual spring constraints.
- `utils/math_utils.py` has quaternion operations and rotation helpers used throughout.
- Contact: Sirui Chen (ericcsr@stanford.edu)
