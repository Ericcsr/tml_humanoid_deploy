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

## Run sim2sim with foot odometer as velocity and position estimator
```
python run_controller.py --use_sim --config exported_policies/sirui_test/experiment.yaml --use_odom
python run_controller.py --use_sim --config exported_policies/yanjie_test/experiment.yaml --use_odom
```

## Some note
1. Sim2Real code remain untested
2. Anchor link is reset to `pelvis` instead of default `torso_link`.
3. If you have any question please contact Sirui Chen `ericcsr@stanford.edu`