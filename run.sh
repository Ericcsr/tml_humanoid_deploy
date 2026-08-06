# conda activate bm
# redis-server --daemonize yes

python run_controller.py \
    --config exported_policies/scenebot_internal/experiment_freespace.yaml \
    --use_sim