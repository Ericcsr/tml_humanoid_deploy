# Kinematic Planner Interface for Local Tracking

## Overview

A closed-loop kinematic planner sits on top of the RL tracking policy (`RLBMPolicy`), replacing the static `.npz` reference motion with real-time generated targets. The planner reacts to robot feedback each tick.

```
robot_state ──> YOUR PLANNER ──> ref motion frame ──> RLBMPolicy ──> ONNX policy ──> joint actions
     ^                                                                                      |
     └──────────────────────────── PD control + physics ────────────────────────────────────┘
```

## Planner Output (per tick, 50 Hz)

Your planner must produce these 4 arrays every 20ms (`control_dt=0.02`):

| Output | Shape | Format | Description |
|--------|-------|--------|-------------|
| `ref_joint_pos` | `(29,)` | float32, **Isaac order** | Target joint angles (rad) |
| `ref_joint_vel` | `(29,)` | float32, **Isaac order** | Target joint velocities (rad/s) |
| `ref_anchor_pos` | `(3,)` | float32, world frame | Pelvis XYZ position |
| `ref_anchor_orn` | `(4,)` | float32, **xyzw** quaternion | Pelvis orientation |

These are consumed by `RLBMPolicy._control_signals_from_motion()` in `rl_policy.py:111-117`.

### How the policy uses them

In `prepare_control_signals()` (`rl_policy.py:119-141`):

1. **`command`** = `[ref_joint_pos, ref_joint_vel]` concatenated (58 values)
2. Anchor pos/orn are transformed to init-relative frame (removing frame-0 heading/position):
   - `rel_anchor_pos = init_root_heading_inv * (ref_anchor_pos - init_root_pos)`
   - `rel_anchor_orn = init_root_heading_inv * ref_anchor_orn`
3. Then transformed to robot body frame:
   - **`motion_anchor_ori_b`** = `robot_orn_inv * rel_anchor_orn`, as flattened 3x2 rotation matrix (6 values)
   - **`projected_gravity`** = `robot_orn_inv * [0,0,-1]` (3 values)

**Note:** `motion_anchor_pos_b` is commented out in local_tracking config — position is not observed by the policy, only orientation.

### Frame-0 anchoring

`init_root_pos` and `init_root_heading_inv` are computed from the **first frame** your planner produces (`rl_policy.py:93-95`). All subsequent frames are relative to this origin. Your planner's first output defines the reference coordinate system.

## Planner Input (robot feedback, available each tick)

| Input | Shape | Format | Source | On real robot? |
|-------|-------|--------|--------|----------------|
| `q` | `(29,)` | **MuJoCo order** | Joint encoders | Yes |
| `dq` | `(29,)` | **MuJoCo order** | Joint encoders | Yes |
| `imu_quat` | `(4,)` | **wxyz** | IMU | Yes |
| `omega` | `(3,)` | world frame | IMU gyro | Yes |
| `root_orn` | `(4,)` | xyzw, heading-relative | Computed from IMU | Yes |
| `last_action` | `(29,)` | Isaac order | Previous policy output | Yes |

With `use_root_state: False` (local tracking config), `root_pos` stays `[0,0,0]` and `root_vel` is unavailable. Only proprioception + IMU orientation are available on the real robot.

### Joint order conversion

Use the index arrays in `utils/params.py` to convert between orderings:
- `q_isaac = q_mujoco[MUJOCO_TO_ISAAC]` (robot → planner/policy)
- `q_mujoco = q_isaac[ISAAC_TO_MUJOCO]` (planner/policy → robot)

## Observation Vector (for reference)

The local_tracking config (`exported_policies/local_tracking/experiment.yaml`) assembles obs in this order:

| obs_name | Size | Source |
|----------|------|--------|
| `command` | 58 | ref_joint_pos(29) + ref_joint_vel(29) |
| `motion_anchor_ori_b` | 6 | flattened 3x2 rotation matrix |
| `projected_gravity` | 3 | gravity in body frame |
| `omega` | 3 | angular velocity |
| `q` | 29 | joints (Isaac order, minus defaults) |
| `dq` | 29 | joint velocities (Isaac order) |
| `last_action` | 29 | previous action |

## Action Pipeline

Policy output → joint commands (`run_controller.py:183-184`):
```
scaled_action = action[ISAAC_TO_MUJOCO] * action_scale + default_joint_pos[ISAAC_TO_MUJOCO]
env.step_robot(scaled_action)  # PD control: tau = kp*(target - q) - kd*dq, clipped to torque_limit
```

## Integration Options

### Option A: Modify RLBMPolicy in-place
Override `_control_signals_from_motion()` to call your planner instead of indexing `.npz` arrays. Minimal changes, keeps observation/action pipeline intact.

### Option B: Separate process via Redis
Planner runs as a separate process, publishes ref motion frames to Redis (similar to `run_state_estimation.py`). Good if planner has different compute requirements or runs on different hardware.

### Option C: New policy subclass
Create `RLPlannerPolicy(RLBMPolicy)` that overrides the motion source. Keeps original code untouched.

## Key Files

| File | What matters |
|------|-------------|
| `rl_policy.py:77-165` | `RLBMPolicy` — ref motion loading, control signals, obs assembly |
| `rl_policy.py:111-117` | `_control_signals_from_motion()` — **the function to replace/override** |
| `run_controller.py:101-186` | Main control loop |
| `run_controller.py:119-125` | `use_root_state=False` path: root_orn from IMU only |
| `exported_policies/local_tracking/experiment.yaml` | Config: obs_names, PD gains, torque limits |
| `utils/params.py` | `ISAAC_TO_MUJOCO`, `MUJOCO_TO_ISAAC`, `ACTION_SCALE` |
| `utils/robot_states.py` | `G1RobotState` dataclass |

## Quaternion Conventions

Be careful — different parts of the codebase use different quaternion orderings:
- **IMU / MuJoCo native:** wxyz (scalar-first)
- **Mocap / Scipy / ref_anchor_orn:** xyzw (scalar-last)
- **body_quat_w in .npz:** wxyz → converted to xyzw at load time (`rl_policy.py:100`)
