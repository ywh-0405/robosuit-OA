# OpenArmX Standalone Hanging Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_openarmx_standalone_viewer.py` start in a hanging dual-arm presentation that keeps the grippers open and leaves the hands visibly away from the floor.

**Architecture:** Keep the change localized to the standalone viewer script by introducing a new `hanging` pose preset, switching the CLI default to that preset, and slightly increasing the default base lift. Protect the behavior with focused script-level tests that verify the new preset, the default argument, and the resulting base height.

**Tech Stack:** Python, `argparse`, `numpy`, `pytest`, robosuite / MuJoCo viewer

---

### Task 1: Add failing tests for the new hanging preset and CLI default

**Files:**
- Modify: `/home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py`
- Test: `/home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_hanging_pose_preset_keeps_grippers_open_and_differs_from_zero():
    module = load_script_module()

    zero_qpos = module.build_initial_qpos("zero")
    hanging_qpos = module.build_initial_qpos("hanging")

    assert len(hanging_qpos) == 18
    assert list(hanging_qpos) != list(zero_qpos)
    assert hanging_qpos[7] == 0.044
    assert hanging_qpos[8] == 0.044
    assert hanging_qpos[16] == 0.044
    assert hanging_qpos[17] == 0.044


def test_main_defaults_to_hanging_pose(monkeypatch):
    module = load_script_module()
    captured = {}

    def fake_build_standalone_sim(pose="zero", base_z_offset=module.DEFAULT_BASE_Z_OFFSET):
        captured["pose"] = pose
        captured["base_z_offset"] = base_z_offset

        class DummyRobot:
            __class__ = type("OpenArmX", (), {})
            root_body = "robot0_openarmx_base"
            joints = list(range(18))
            eef_name = {"right": "gripper0_right", "left": "gripper0_left"}

        class DummySim:
            model = type("_Model", (), {"_model": object()})()
            data = type("_Data", (), {"_data": object()})()

            def forward(self):
                return None

        return DummySim(), DummyRobot()
```

- [ ] **Step 2: Finish the CLI default test body**

```python
    class DummyViewerContext:
        cam = type("Cam", (), {})()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return False

        def sync(self):
            return None

    monkeypatch.setattr(module, "build_standalone_sim", fake_build_standalone_sim)
    monkeypatch.setattr(module.viewer, "launch_passive", lambda *args, **kwargs: DummyViewerContext())
    monkeypatch.setattr(module, "configure_camera", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_openarmx_standalone_viewer.py"])

    module.main()

    assert captured["pose"] == "hanging"
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pytest /home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py -q`

Expected: FAIL because `build_initial_qpos("hanging")` is unsupported and the parser default is still `zero`.

- [ ] **Step 4: Record the red result**

Expected failures to observe:

```text
ValueError: Unsupported pose preset: hanging
AssertionError: assert 'zero' == 'hanging'
```

- [ ] **Step 5: Skip commit**

This directory is not a git repository, so do not add a commit step. Move directly to implementation after the failing tests are confirmed.

### Task 2: Implement the hanging preset and default standalone presentation

**Files:**
- Modify: `/home/y/Desktop/openarmx_robosuite_example/run_openarmx_standalone_viewer.py`
- Test: `/home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py`

- [ ] **Step 1: Add the hanging pose preset and raise the default base offset**

```python
DEFAULT_BASE_Z_OFFSET = 0.12


def build_initial_qpos(pose):
    if pose == "task":
        return np.array(
            [
                -2.05223,
                0.03461,
                0.01063,
                0.0,
                0.00201,
                -0.00173,
                0.11128,
                0.044,
                0.044,
                1.82583,
                -0.00426,
                -0.11421,
                0.79909,
                0.06046,
                0.15543,
                -1.08681,
                0.044,
                0.044,
            ]
        )
    if pose == "hanging":
        return np.array(
            [
                -1.5708,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.044,
                0.044,
                1.5708,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.044,
                0.044,
            ]
        )
```

- [ ] **Step 2: Switch the CLI default from `zero` to `hanging`**

```python
    parser.add_argument(
        "--pose",
        choices=["zero", "hanging", "task"],
        default="hanging",
        help="Robot pose preset. 'hanging' is the default display pose; 'zero' is better for inspection; 'task' matches the task-oriented robosuite default.",
    )
```

- [ ] **Step 3: Keep the zero preset unchanged for inspection use**

```python
    if pose == "zero":
        qpos = np.zeros(18)
        qpos[7] = qpos[8] = qpos[16] = qpos[17] = 0.044
        return qpos
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest /home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py -q`

Expected: PASS with all standalone-viewer tests green.

- [ ] **Step 5: Skip commit**

This directory is not a git repository, so do not add a commit step. Continue to full verification.

### Task 3: Verify the full script-level regression coverage

**Files:**
- Test: `/home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py`
- Test: `/home/y/Desktop/openarmx_robosuite_example/test_openarmx_viewer_script.py`

- [ ] **Step 1: Run both script test files**

Run: `pytest /home/y/Desktop/openarmx_robosuite_example/test_openarmx_standalone_viewer_script.py /home/y/Desktop/openarmx_robosuite_example/test_openarmx_viewer_script.py -q`

Expected: PASS with no regressions in the standalone viewer or the two-arm viewer script checks.

- [ ] **Step 2: Confirm the base-height assertion still holds**

The existing test must still verify:

```python
base_id = sim.model.body_name2id("robot0_openarmx_base")
assert sim.data.body_xpos[base_id][2] > 0.05
```

If the implementation changes the practical threshold, update the assertion only if needed and only to reflect the new intended visual height.

- [ ] **Step 3: Summarize the outcome with evidence**

Report:

- the new default pose name
- the default base offset value
- the exact `pytest` commands that passed

- [ ] **Step 4: Skip commit**

This directory is not a git repository, so end with verified file changes only.
