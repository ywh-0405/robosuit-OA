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
python run_openarmx_standalone_viewer.py --approach-height 0.10 --grasp-height-offset 0.008 --lift-height 0.18
python run_openarmx_standalone_viewer.py --fingertip-backoff 0.012 --table-clearance 0.006
python run_openarmx_standalone_viewer.py --ik-iters 10 --ik-gain 0.28 --ik-max-step 0.03
python run_openarmx_standalone_viewer.py --orientation-weight 0.0 --grasp-yaw 0.0 --close-steps 45 --settle-steps 60
python run_openarmx_standalone_viewer.py --debug-state
python run_openarmx_standalone_viewer.py --assisted-lift --debug-state
python run_openarmx_standalone_viewer.py --pure-physics --debug-state
python run_openarmx_standalone_viewer.py --scripted-grasp
```

Notes:

- Default mode uses a staged small-object grasp with MuJoCo physics enabled: align x/y over the cube center first, approach by position, descend, close, settle, then lift.
- `--pose right_grasp` is the default starting pose for the right-arm IK solver.
- `--base-z` defaults to `0.4`, so the robot is lifted above the table.
- The cube is placed on the tabletop by `--cube-x`, `--cube-y`, `--cube-z`; table top is `z=0.8`.
- `--grasp-height-offset` defaults to `0.008`, so the gripper closes around the upper half of the cube instead of pressing down into the table.
- The IK target is computed from explicit fingerpad collision boxes added to the right gripper at runtime, not from a guessed TCP offset.
- The gripper closes each finger to half of the desired inner jaw gap. This matters because OpenArmX has two independent slide joints; commanding the total gap to each finger leaves the jaw twice too open.
- Default mode is a visual centered-grasp demo: once close / settle / lift begins, the cube center is placed at the center of the two right fingerpads and then follows that center upward during lift.
- `--no-assisted-lift` is kept as a legacy flag for old commands and still runs the visual centered-grasp demo. Use `--pure-physics` when you deliberately want to turn off visual cube centering and inspect raw contact behavior.
- A strict `--pure-physics` lift is not reliable yet with the current OpenArmX finger collision geometry and kinematic IK qpos stepping: very small cubes can contact the fingers but still slip instead of lifting consistently.
- IsaacLab's minimal OpenArmX scripts are not directly comparable to this raw-contact lift test: they load a USD articulation and drive joints through implicit PD actuators, while this robosuite demo computes IK and writes joint qpos directly.
- Use `--no-step-physics` only for static inspection. Real contact grasping needs the default physics stepping.
- If the fingers still touch the table, try `--grasp-height-offset 0.012 --table-clearance 0.01`.
- If the gripper still nudges the cube sideways, try `--ik-gain 0.20 --ik-max-step 0.005`.
- If the gripper approaches from the wrong yaw, try `--orientation-weight 0.08 --grasp-yaw 1.5708`.
- Use `--scripted-grasp` only for the older visual demo that keeps the cube centered between the gripper fingers.

## Expected Result

The script should print:

- environment type: `TwoArmLift`
- robot list including `OpenArmX`
- action dimension: `16`

For the viewer script:

- default behavior uses zero action, so you can inspect the initial pose steadily
- `--random-action` makes the robot and pot move around for a quick visual check

## Collect Data And Train RL

This pipeline still uses your OpenArmX dual-arm robot model. The current first training target controls the right arm / right gripper for the small-cube grasp, while the left arm is parked open. The UR5e thesis project is only used as the reference pattern for data format, replay buffer, DDPG+BC training, and evaluation.

Collect visual centered-grasp expert data:

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example
python collect_openarmx_data.py --episodes 20 --max-steps 220
```

This writes compressed episodes to `openarmx_visual_grasp_dataset/`. Each `.npz` contains `obs`, `actions`, `rewards`, `next_obs`, `dones`, `success`, and `episode_length`, matching the structure used by the UR5e reference training code.

Train DDPG+BC from those demonstrations:

```bash
python train_openarmx_ddpg_bc.py \
  --dataset-dir openarmx_visual_grasp_dataset \
  --checkpoint checkpoints/openarmx_visual_actor.pth \
  --pretrain-steps 200 \
  --total-steps 1000
```

PyTorch must be installed inside the `robosuit` conda environment before training. The current environment can collect data, but training will stop with a clear PyTorch-required message if `torch` is missing.

Evaluate a trained actor:

```bash
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor.pth \
  --episodes 5
```

Use `--render` on evaluation only when you want a headed MuJoCo viewer. For bulk data collection and training, keep rendering off so the run does not appear stuck behind a GUI window.

### Random Cube Position Training

The first stable random-position range is intentionally small:

- cube x: `-0.305` to `-0.270`
- cube y: `-0.160` to `-0.120`

This range was chosen because the current right-arm IK expert succeeds reliably there. Wider ranges can be added later as a curriculum.

Collect random-position expert data:

```bash
python collect_openarmx_data.py \
  --output-dir openarmx_visual_grasp_dataset_random_tight \
  --episodes 50 \
  --max-steps 220 \
  --random-cube-pos
```

Train on the random-position dataset and online fine-tune in the same random distribution:

```bash
python train_openarmx_ddpg_bc.py \
  --dataset-dir openarmx_visual_grasp_dataset_random_tight \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --pretrain-steps 300 \
  --total-steps 1500 \
  --random-cube-pos
```

Evaluate random-position success rate:

```bash
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --episodes 50 \
  --max-steps 220 \
  --random-cube-pos
```

If you later want, I can also add:

- an on-screen renderer version
- a random-action demo
- a keyboard / teleop example
