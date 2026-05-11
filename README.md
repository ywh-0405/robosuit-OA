# OpenArmX robosuite Example

This folder contains a minimal example showing how to use `OpenArmX` as a built-in robosuite robot in the `robosuit` conda environment.

## Files

- `run_openarmx_twoarmlift.py`: creates a `TwoArmLift` environment with `robots="OpenArmX"` and prints a few basic properties.
- `run_openarmx_twoarmlift_viewer.py`: opens the MuJoCo viewer so you can watch `OpenArmX` inside `TwoArmLift`.
- `run_openarmx_standalone_viewer.py`: shows `OpenArmX` in a small tabletop cube grasping scene.

## Run

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example
python run_openarmx_twoarmlift.py
```

## Visualize

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example
python run_openarmx_twoarmlift_viewer.py
```

Optional:

```bash
python run_openarmx_twoarmlift_viewer.py --random-action
python run_openarmx_twoarmlift_viewer.py --camera-id 0 --steps 2000
```

## Visualize Small Block Grasp

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example
python run_openarmx_standalone_viewer.py
```

Optional:

```bash
python run_openarmx_standalone_viewer.py --pose zero
python run_openarmx_standalone_viewer.py --pose right_grasp --base-z 0.4
python run_openarmx_standalone_viewer.py --cube-x -0.30 --cube-y -0.14 --cube-z 0.815
python run_openarmx_standalone_viewer.py --approach-height 0.16 --grasp-height-offset 0.008 --lift-height 0.18
python run_openarmx_standalone_viewer.py --fingertip-backoff 0.012 --table-clearance 0.006
python run_openarmx_standalone_viewer.py --ik-iters 3 --ik-gain 0.28 --ik-max-step 0.008
python run_openarmx_standalone_viewer.py --orientation-weight 0.22 --grasp-yaw 0.0
python run_openarmx_standalone_viewer.py --assisted-lift
python run_openarmx_standalone_viewer.py --scripted-grasp
```

Notes:

- Default mode uses a slow UR5e-style staged grasp with MuJoCo physics enabled: align x/y over the cube center first, rotate the gripper top-down, descend vertically, close slowly, then lift.
- `--pose right_grasp` is the default starting pose for the right-arm IK solver.
- `--base-z` defaults to `0.4`, so the robot is lifted above the table.
- The cube is placed on the tabletop by `--cube-x`, `--cube-y`, `--cube-z`; table top is `z=0.8`.
- `--grasp-height-offset` defaults to `0.008`, so the gripper closes around the upper half of the cube instead of pressing down into the table.
- The IK target is computed from the two finger collision meshes, not from a guessed TCP offset.
- Default mode does not attach the cube to the gripper; misses stay visible. Use `--assisted-lift` only as an explicit old-style debug aid.
- Use `--no-step-physics` only for static inspection. Real contact grasping needs the default physics stepping.
- If the fingers still touch the table, try `--grasp-height-offset 0.012 --table-clearance 0.01`.
- If the gripper still nudges the cube sideways, try `--ik-gain 0.20 --ik-max-step 0.005`.
- If the gripper approaches from the wrong yaw, try `--grasp-yaw 1.5708`.
- Use `--scripted-grasp` only for the older visual demo that keeps the cube centered between the gripper fingers.

## Expected Result

The script should print:

- environment type: `TwoArmLift`
- robot list including `OpenArmX`
- action dimension: `16`

For the viewer script:

- default behavior uses zero action, so you can inspect the initial pose steadily
- `--random-action` makes the robot and pot move around for a quick visual check

If you later want, I can also add:

- an on-screen renderer version
- a random-action demo
- a keyboard / teleop example
