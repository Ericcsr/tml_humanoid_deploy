# VLA motion bridge

Drives real-robot motion tracking from a **VLA (GR00T) policy** running on a
separate server. The bridge turns GR00T reference-motion chunks into the live
Redis streams that `run_controller.py`'s `RLStreamingContactPolicy` already
consumes, and feeds proprioception + world root state back to the VLA server as
observations.

Target checkpoint: the **heading + sin/cos, pelvis-stereo** variant —
`vlk_g1_contact_fk_heading_sincos_pelvis_config` /
`groot_ckpt/vlk_g1_v1_heading_sincos/vlk_apartment_noforce_nofk_lowres_gma_p`.

```
STATE  = joint_pos_sincos(58) joint_vel(29) root_dpos_local(3) root_dyaw(1)
         root_deheaded_rot6d(6) contact(10)  + pelvis_left_view pelvis_right_view
ACTION = joint_pos_sincos(58) contact(10) root_dpos_local(3) root_dyaw(1)
         root_deheaded_rot6d(6)             over a 25-step horizon
```

## Architecture

```
 [superodometry]──lidar──▶[run_state_estimation_slam_only.py --use_imu]
                              proprio_data ─▶ root_data            (NOT runnable here)
                                                │
 [run_controller.py --use_odom --config .../experiment_stair_streaming.yaml]
   writes proprio_data, reads root_data, RLStreamingContactPolicy
        ▲ 6 pub/sub channels                    │
        │                                        │
 [vla_motion_bridge.py]  (bm env, MuJoCo FK)     │
   reads root_data + proprio_data + camera:* ────┘
   packs vla:obs ──▶  [stair_vla_server.py]  (rendering env, GR00T, GPU)
   polls vla:action ◀──
   → decode sin/cos, integrate heading deltas from the CURRENT root anchor,
     MuJoCo-FK the 3-point poses, publish the 6 channels @ 50 Hz w/ replanning
```

The bridge is the only new runtime component. It replaces the controller↔sim glue
that `stair-rendering/stair_sim_eval.py` does in-process, split across the real
Redis boundary. The heading+sincos integration/decode/FK math is ported verbatim
from that file's **sync** path (its async path raises `NotImplementedError` for
this representation) — see `heading_sincos.py` and `vr3_fk.py`.

## Files

| File | Role |
|---|---|
| `vla_motion_bridge.py` | The bridge (obs↔chunk↔6-channel streamer). |
| `heading_sincos.py` | Heading+sincos encode/decode + chunk integration (ported). |
| `vr3_fk.py` | MuJoCo FK of the wrist/torso 3-point poses (ported `fk_vr3`). |
| `../exported_policies/stair_assets/experiment_stair_streaming.yaml` | `use_streaming_motion: True` controller config. |
| `test_roundtrip.py` | In-process math check (no Redis): encode→integrate→FK reproduces a clip. |
| `test_consumer.py` | Drives the real `RLStreamingContactPolicy` against the live bridge stream. |
| `mocks/mock_robot_state.py` | Fakes SLAM `root_data` + `proprio_data` by replaying a clip. |
| `mocks/mock_camera.py` | Publishes JPEG frames to `camera:*` topics. |
| `mocks/mock_vla_server.py` | GPU-free GR00T stand-in: replays a clip as heading+sincos chunks. |
| `mocks/sim_gt_publisher.py` | `--use_sim` estimator replacement: GT `root_data` from the sim's SharedMemory (no SLAM). |
| `mocks/train_frame_camera.py` | Serves a fixed training-set frame (default #20) on the pelvis camera topics. |
| `run_mock_harness.sh` | Launches all mocks + the bridge together (bench test, no controller). |
| `run_sim_dryrun.sh` | **Full sim dry run**: real controller (`--use_sim`) + GT root + frame-20 cameras + GR00T + bridge. |

## Camera contract

The (not-yet-built) camera server must publish **raw JPEG bytes** to latest-wins
Redis STRING keys, one per GR00T video key. Defaults:
`camera:pelvis_left_view`, `camera:pelvis_right_view`. Remap with repeated
`--video_topic KEY=TOPIC` on the bridge.

## Local test (no robot / GPU / lidar)

```bash
conda activate bm
redis-server --daemonize yes            # if not already up

# 1) Pure-math correctness (fast, no processes):
python vla_bridge/test_roundtrip.py \
  --clip ../motion_data/dataset_real_pose_filtered_bm_norm/up/h0.190_w0.250_s0.4_t0.npz

# 2) Full loop with mocks:
bash vla_bridge/run_mock_harness.sh
#    then, in another shell, either inspect the channels...
python target_server/stream_subscriber.py --expect_contact_dim 10
#    ...or drive the REAL streaming policy against the stream:
python vla_bridge/test_consumer.py
```

## Simulation dry run (no robot / SLAM / lidar; GPU used for real GR00T)

Runs the WHOLE pipeline against the MuJoCo sim. MuJoCo is the ground truth, so
`sim_gt_publisher.py` supplies `root_data` directly (no estimator, no
superodometry — as requested for `--use_sim`), and the cameras are frozen to the
training set's 20th frame.

```bash
conda activate bm
bash vla_bridge/run_sim_dryrun.sh                 # real GR00T (rendering env, GPU)
bash vla_bridge/run_sim_dryrun.sh --mock_vla      # GPU-free clip-replay stand-in
```

The controller runs in the **foreground**: a MuJoCo viewer opens and
`maintain_state()` waits for you to press **Enter** before the robot moves. All
other processes run in the background with logs in `/tmp/vla_dryrun/`. Ctrl-C
tears the whole stack down.

Note: `experiment_stair_streaming.yaml` carries a `ref_motion_path` used ONLY to
seat the robot on the stairs at sim startup (`--use_sim`); the streaming policy
never loads it and it is unused on the real robot.

## Real deployment

```bash
# superodometry host:
start_slam.sh
# deploy host:
python run_state_estimation_slam_only.py --use_imu
python run_controller.py --net <iface> --use_odom \
  --config exported_policies/stair_assets/experiment_stair_streaming.yaml
# GPU host (rendering env):
python stair_vla_server.py --model_path <heading_sincos_ckpt> --embodiment_tag new_embodiment
# deploy host (bm env):
python vla_bridge/vla_motion_bridge.py --task "walk up the stairs"
# + the camera server publishing camera:pelvis_left_view / camera:pelvis_right_view
```

## Conventions (must hold across the boundary)

- Quaternions stored **wxyz**; scipy in = xyzw (`[[1,2,3,0]]`), out = wxyz
  (`[[3,0,1,2]]`). `root_data`/`G1RobotState` root_orn are **xyzw**.
- Joint order: GR00T obs/action = **ISAAC**; `proprio_data`/MuJoCo = **MuJoCo**;
  remap via `ISAAC_TO_MUJOCO` / `MUJOCO_TO_ISAAC`.
- `joint_pos_sincos` = contiguous `[sin(29), cos(29)]`, ISAAC order.
- `root_deheaded_rot6d` is **absolute** tilt (heading removed), not a delta.
- Contact: GR00T emits 10-way; the streaming policy keeps `arr[:8] + zeros(2)`.
- The heading chunk is re-anchored to the robot's **current** `root_data` each
  time an obs is sent — no cross-chunk world-frame drift.
```
