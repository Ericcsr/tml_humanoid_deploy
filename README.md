# TML Humanoid Deploy

A deployment framework for humanoid robot control supporting both MuJoCo simulation and real robot hardware.

## Installation

```bash
Install all dependencies
Then install the package with
pip install -e .

```

## Usage

### MuJoCo Simulation
```bash
python scripts/mujoco_launch.py --config exported_policies/sirui_test/experiment.yaml
OR
python mujoco_launch --config exported_policies/sirui_test/experiment.yaml

```

### With Odometry
```bash
python scripts/mujoco_launch.py --config exported_policies/sirui_test/experiment.yaml --use_odom
```

### Real Robot
```bash
python scripts/real_launch.py --config exported_policies/sirui_test/experiment.yaml --net <network_interface>
OR
real_launch --config exported_policies/sirui_test/experiment.yaml --net <network_interface>

```

## Configuration

Experiment configurations are stored in `exported_policies/`. Each experiment folder contains:
- `experiment.yaml` - Configuration file
- `policy.onnx` - Trained policy model
- `motion.npz` - Reference motion data

## Notes

1. Anchor link is reset to `pelvis` instead of default `torso_link`
2. For questions, contact Sirui Chen `ericcsr@stanford.edu`