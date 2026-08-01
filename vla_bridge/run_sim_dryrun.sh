#!/usr/bin/env bash
# ============================================================================
# VLA control pipeline — SIMULATION DRY RUN (--use_sim, ground-truth root).
#
# Brings up the WHOLE stack on one machine, no robot / no superodometry / no
# SLAM estimator. MuJoCo is the ground truth: sim_gt_publisher feeds GT root_data
# straight to Redis, and the cameras are frozen to the training set's 20th frame.
#
#   run_controller.py --use_sim --use_odom   (MuJoCo sim + tracking policy)   [FG]
#   sim_gt_publisher.py                       (GT root_data <- sim shm)        [bg]
#   train_frame_camera.py                     (frame-20 pelvis JPEGs)          [bg]
#   stair_vla_server.py                       (real GR00T, rendering env, GPU) [bg]
#   vla_motion_bridge.py                      (obs<->chunk<->6 channels)       [bg]
#
# The controller runs in the FOREGROUND because (a) it opens the MuJoCo viewer
# and (b) maintain_state() waits for you to press Enter before the robot moves.
# Everything else runs in the background; logs go to /tmp/vla_dryrun/.
#
# Usage:
#   bash vla_bridge/run_sim_dryrun.sh [--mock_vla] [--ckpt DIR] [--task STR]
#     --mock_vla   use the GPU-free clip-replay server instead of real GR00T
#     --ckpt DIR   GR00T checkpoint (default: heading+sincos pelvis, checkpoint-20000)
#     --task STR   language instruction (default "walk up the stairs")
# Ctrl-C (or exiting the controller) tears the whole stack down.
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$HERE")"
PROJECTS="$(dirname "$DEPLOY_ROOT")"
cd "$DEPLOY_ROOT"

# ── Config (override via flags / env) ──
PY_BM="${PY_BM:-/home/ubuntu/miniconda3/envs/bm/bin/python}"
PY_RENDER="${PY_RENDER:-/home/ubuntu/miniconda3/envs/rendering/bin/python}"
CONFIG="exported_policies/stair_assets/experiment_stair_streaming.yaml"
CKPT="${STAIR_CKPT:-$PROJECTS/groot_ckpt/vlk_g1_v1_heading_sincos/vlk_apartment_noforce_nofk_lowres_gma_p/checkpoint-20000}"
GR00T_ROOT="${GR00T_ROOT:-$PROJECTS/Isaac-GR00T}"
TASK="walk up the stairs"
USE_MOCK_VLA=0
CLIP="$PROJECTS/motion_data/dataset_real_pose_filtered_bm_norm/up/h0.190_w0.250_s0.4_t0.npz"
LOGDIR="/tmp/vla_dryrun"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock_vla) USE_MOCK_VLA=1; shift ;;
    --ckpt)     CKPT="$2"; shift 2 ;;
    --task)     TASK="$2"; shift 2 ;;
    --clip)     CLIP="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$LOGDIR"
echo "[dryrun] deploy_root = $DEPLOY_ROOT"
echo "[dryrun] config      = $CONFIG"
echo "[dryrun] logs        = $LOGDIR"
[[ $USE_MOCK_VLA -eq 1 ]] && echo "[dryrun] VLA server  = MOCK (clip replay: $(basename "$CLIP"))" \
                          || echo "[dryrun] VLA server  = REAL GR00T ($CKPT)"

# ── Redis ──
if ! redis-cli ping >/dev/null 2>&1; then
  echo "[dryrun] starting redis-server ..."
  redis-server --daemonize yes >/dev/null 2>&1 || { echo "cannot start redis"; exit 1; }
  sleep 1
fi
redis-cli flushdb >/dev/null 2>&1
echo "[dryrun] redis ready."

# ── Background process bookkeeping + teardown ──
declare -a PIDS=()
launch() {  # launch "name" logfile cmd...
  local name="$1" log="$2"; shift 2
  echo "[dryrun] launching $name  (log: $log)"
  ( "$@" ) >"$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  echo "         pid=$pid"
}
cleanup() {
  echo; echo "[dryrun] tearing down ..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  echo "[dryrun] done."
}
trap cleanup EXIT INT TERM

# ── 1) VLA server (real GR00T in rendering env, or the GPU-free mock) ──
if [[ $USE_MOCK_VLA -eq 1 ]]; then
  launch "mock_vla_server" "$LOGDIR/vla_server.log" \
    "$PY_BM" vla_bridge/mocks/mock_vla_server.py --clip "$CLIP"
else
  launch "stair_vla_server (GR00T)" "$LOGDIR/vla_server.log" \
    env GR00T_ROOT="$GR00T_ROOT" "$PY_RENDER" \
    "$PROJECTS/stair-rendering/stair_vla_server.py" \
    --model_path "$CKPT" --embodiment_tag new_embodiment \
    --task "$TASK" --redis_ns vla --gr00t_root "$GR00T_ROOT"
fi

# ── 2) Fixed training-frame cameras (pelvis stereo, frame 20) ──
launch "train_frame_camera" "$LOGDIR/camera.log" \
  "$PY_BM" vla_bridge/mocks/train_frame_camera.py --frame_index 20

# ── 3) Wait for the VLA server to be ready (real GR00T load can take minutes) ──
echo "[dryrun] waiting for vla:status == ready (see $LOGDIR/vla_server.log) ..."
for i in $(seq 1 900); do
  [[ "$(redis-cli get vla:status 2>/dev/null)" == "ready" ]] && { echo "[dryrun] VLA server ready."; break; }
  sleep 1
  if [[ $i -eq 900 ]]; then echo "[dryrun] WARNING: VLA server not ready after 900s; continuing anyway."; fi
done

# ── 4) sim_gt_publisher — waits for the sim's shared memory, then feeds root_data.
#       Safe to start now: it retries attach until run_controller spawns the sim. ──
launch "sim_gt_publisher" "$LOGDIR/sim_gt.log" \
  "$PY_BM" vla_bridge/mocks/sim_gt_publisher.py --hz 200 --attach_timeout 300

# ── 5) The bridge (obs -> VLA -> chunk -> 6 controller channels) ──
launch "vla_motion_bridge" "$LOGDIR/bridge.log" \
  "$PY_BM" vla_bridge/vla_motion_bridge.py --task "$TASK" --server_timeout 900

echo
echo "============================================================================"
echo "[dryrun] Background stack is up. Launching the CONTROLLER in the foreground."
echo "[dryrun] A MuJoCo viewer window will open. When prompted"
echo "[dryrun]   'Press Enter to continue...'  press ENTER here to release the robot."
echo "[dryrun] Tail background logs in another shell:  tail -F $LOGDIR/*.log"
echo "[dryrun] Ctrl-C to stop EVERYTHING."
echo "============================================================================"
echo

# ── 6) Controller (foreground: viewer + the maintain_state Enter gate) ──
"$PY_BM" run_controller.py --use_sim --use_odom --config "$CONFIG"

echo "[dryrun] controller exited."
