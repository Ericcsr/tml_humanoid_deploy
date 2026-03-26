# TML Humanoid Deploy

## Example experiment folder:
```
- experiment_folder_name
    - experiment.yaml (copy and extend from existing experiments)
    - policy.onnx (obtained from logs/rsl_rl/... in training code base)
    - ref_motion.npz (obtained from artifacts in training code base)
```

## Run sim2sim with privilege info
```
python run_controller.py --use_sim --config exported_policies/sirui_test/experiment.yaml
```

## Run sim2real(sim) with foot odometer as velocity and position estimator
```
# Terminal 1
python run_controller.py (--use_sim) --config exported_policies/sirui_test/experiment.yaml --use_odom
# Terminal 2
python run_state_estimation.py --use_sim --visualize
```
Note sim2real requires running lidar slam code [link](https://github.com/Ericcsr/G1_localization)
## Run sim2real(sim) with SLAM fusing with foot odom
```
python run_controller.py (--use_sim) --config exported_policies/sirui_test/experiment.yaml --use_odom --use_slam
python run_state_estimation.py --use_sim --visualize --use_slam --use_acc
python acc_estimator_mp.py
```


## Some note
1. Sim2Real code remain untested
2. Anchor link is reset to `pelvis` instead of default `torso_link`.
3. If you have any question please contact Sirui Chen `ericcsr@stanford.edu`
