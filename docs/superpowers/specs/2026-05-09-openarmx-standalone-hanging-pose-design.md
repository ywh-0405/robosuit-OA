# OpenArmX Standalone Hanging Pose Design

## Goal

Make `run_openarmx_standalone_viewer.py` open with a default robot presentation that looks clearly separated from the floor, with both arms in a vertical hanging posture and grippers open.

The visual target is the same distance impression the user sees in `run_openarmx_twoarmlift_viewer.py`, but applied to the standalone empty-arena viewer where there is no table.

## Scope

Only update `run_openarmx_standalone_viewer.py` and its tests.

Do not change:

- `run_openarmx_twoarmlift_viewer.py`
- task environment behavior
- camera controls beyond keeping the current default view compatible with the new pose

## Design

### Pose preset

Add a new pose preset named `hanging`.

Requirements:

- it should not be the all-zero inspection pose
- both arms should be configured for a natural-looking downward hanging presentation
- both grippers remain open at the same open value currently used by `zero`

### Default behavior

Change the standalone viewer default pose from `zero` to `hanging`.

This makes the script show the requested presentation without extra CLI arguments.

### Base height

Increase the default `base_z_offset` enough that the hanging hands are visibly away from the floor.

The exact value should be chosen empirically in code with a regression test that confirms the robot base remains clearly above the floor.

## Testing

Add or update tests to verify:

- `build_initial_qpos("hanging")` exists and returns 18 joints
- `hanging` keeps both grippers open
- `hanging` is distinct from `zero`
- the CLI default pose is `hanging`
- the standalone sim still places the robot base above the floor

## Risks

- If the hanging joint values are too aggressive, the pose may look unnatural or cause self-collision.
- If the base lift is too small, the visual result will still look cramped.
- If the base lift is too large, the robot may look disconnected from the scene.

## Non-Goals

- matching an exact `TwoArmLift` joint configuration
- adding IK or dynamic pose generation
- changing the viewer interaction model
