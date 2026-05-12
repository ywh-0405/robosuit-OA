# OpenArmX RL Visual Grasp Design

## Goal

Build a minimal reinforcement-learning pipeline for the current OpenArmX standalone small-cube scene:

1. collect expert demonstration data,
2. train an actor with behavior cloning plus DDPG-style fine tuning,
3. evaluate the actor in the same scene.

This first version targets the already-working visual centered-grasp demo, not strict pure-physics grasp success.

## Reference

The design follows the useful parts of `/home/y/Desktop/UR5e_robosuit_thesis`:

- `collect_data/collect_data_noversion.py`: scripted expert saves `.npz` episodes with `obs`, `actions`, `rewards`, `next_obs`, and `dones`.
- `train/train_final.py`: replay buffer loads `.npz`, does BC pretraining, then DDPG updates.
- `evaluate/evaluate.py`: loads the actor, runs episodes, and reports success rate and reward.

The OpenArmX implementation should not copy the UR5e `Lift` environment directly. It should reuse the current standalone scene from `run_openarmx_standalone_viewer.py`, because that scene contains the fingerpad geometry, close-gap fix, and visual grasp behavior that the user confirmed as the good version.

## Scope

In scope:

- A reusable lightweight OpenArmX visual-grasp environment.
- A deterministic expert policy that uses the same staged grasp behavior as the viewer.
- `.npz` demonstration collection compatible with the UR5e training pattern.
- A compact DDPG+BC trainer for vector observations and bounded continuous actions.
- A compact evaluator.
- Tests for environment shapes, data format, replay loading, and actor output range.

Out of scope for this first version:

- Strict pure-physics training as the default objective.
- Image-based policy learning.
- Long training runs inside tests.
- Uploading datasets or checkpoints to Git.

## Environment Design

Create `openarmx_rl/env.py` with `OpenArmXVisualGraspEnv`.

The environment wraps `build_standalone_sim()` from `run_openarmx_standalone_viewer.py`.

Observation is a 1-D float32 vector:

- right arm joint qpos: 7 values,
- right gripper qpos: 2 values,
- cube position: 3 values,
- fingerpad center: 3 values,
- cube minus fingerpad center: 3 values,
- current expert target position: 3 values,
- current gripper target: 1 value,
- normalized step counter: 1 value,
- grasp state one-hot for `center_xy`, `approach`, `descend`, `close`, `settle`, `lift`: 6 values.

Total observation size: 29.

Action is a 4-D float32 vector in `[-1, 1]`:

- `action[0:3]`: delta applied to the expert target position,
- `action[3]`: delta applied to the expert gripper qpos.

The environment converts action to:

- IK target position = expert target + delta scaled by `action_pos_scale`,
- gripper qpos = expert gripper target + delta scaled by `action_gripper_scale`.

During `close`, `settle`, and `lift`, if visual grasp is enabled, the cube center follows the fingerpad center. This matches the confirmed-good viewer behavior.

## Reward And Success

Reward should be shaped but simple:

- penalize distance between cube and fingerpad center,
- reward lifted cube height above the initial cube height,
- add a success bonus when the cube is visually held and lifted.

Success condition:

- `cube_z >= initial_cube_z + success_lift_height`,
- and cube/fingerpad distance is below `success_center_tolerance`.

Default thresholds:

- `success_lift_height = 0.12`,
- `success_center_tolerance = 0.02`.

## Data Collection

Create `collect_openarmx_data.py`.

Default behavior:

- save to `openarmx_visual_grasp_dataset/`,
- collect a small default number of episodes,
- render off by default,
- save one compressed `.npz` per episode.

Each file stores:

- `obs`,
- `actions`,
- `rewards`,
- `next_obs`,
- `dones`,
- `success`,
- `episode_length`.

The expert action is `[0, 0, 0, 0]` because the environment already uses the deterministic staged expert target as its nominal command. Optional noise can be added for broader data coverage.

## Training

Create `openarmx_rl/ddpg_bc.py` for reusable training pieces:

- `ReplayBuffer`,
- `Actor`,
- `Critic`,
- `DDPGAgent`,
- `load_dataset`.

Create `train_openarmx_ddpg_bc.py` for CLI training.

Default first-run settings should be small enough to smoke test:

- `pretrain_steps`: 200,
- `total_steps`: 1000,
- `batch_size`: 128,
- save actor to `checkpoints/openarmx_visual_actor.pth`.

The trainer should allow larger values through CLI flags.

## Evaluation

Create `evaluate_openarmx_policy.py`.

It loads the actor, runs episodes, and prints:

- mean reward,
- mean episode length,
- success rate.

Rendering is optional with `--render`. Headed render uses the same MuJoCo viewer path as the environment.

## Testing

Tests should not run long training.

Add tests that verify:

- environment reset returns the expected observation shape,
- environment step returns `(obs, reward, done, truncated, info)` with stable shapes and keys,
- collector utility writes `.npz` with the expected fields,
- replay buffer loads that `.npz`,
- actor output is bounded to `[-1, 1]`.

Run verification with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 conda run -n robosuit pytest -q
```

## Documentation And Git Hygiene

Update `README.md` with:

- how to collect data,
- how to train,
- how to evaluate.

Update `.gitignore` so generated datasets and checkpoints are not committed:

- `openarmx_visual_grasp_dataset/`,
- `checkpoints/`,
- `*.pth`.
