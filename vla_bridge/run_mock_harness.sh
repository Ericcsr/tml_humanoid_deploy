#!/usr/bin/env bash
# Launch the full VLA-motion-bridge MOCK harness on this machine (no robot, no
# GPU, no superodometry). Exercises the entire obs -> VLA -> chunk -> 6-channel
# reference-motion loop, then leaves it running so you can attach the real
# RLStreamingContactPolicy consumer (test_consumer.py) or stream_subscriber.py.
#
# Usage:
#   bash vla_bridge/run_mock_harness.sh [CLIP.npz]
# Ctrl-C to stop everything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$HERE")"
cd "$DEPLOY_ROOT"

PY="${PY:-/home/ubuntu/miniconda3/envs/bm/bin/python}"
CLIP="${1:-../motion_data/dataset_real_pose_filtered_bm_norm/up/h0.190_w0.250_s0.4_t0.npz}"

echo "[harness] deploy_root=$DEPLOY_ROOT"
echo "[harness] clip=$CLIP"
redis-cli flushdb >/dev/null 2>&1 || { echo "redis-server not running"; exit 1; }

pids=()
cleanup() { echo; echo "[harness] stopping..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$PY" vla_bridge/mocks/mock_robot_state.py --clip "$CLIP" --hz 50 &
pids+=($!)
"$PY" vla_bridge/mocks/mock_camera.py --hz 30 &
pids+=($!)
"$PY" vla_bridge/mocks/mock_vla_server.py --clip "$CLIP" &
pids+=($!)
sleep 2
"$PY" vla_bridge/vla_motion_bridge.py --server_timeout 30 &
pids+=($!)

echo "[harness] running. In another shell:"
echo "  $PY target_server/stream_subscriber.py --expect_contact_dim 10"
echo "  $PY vla_bridge/test_consumer.py"
wait
