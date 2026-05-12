# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260509-002] correction

**Logged**: 2026-05-09T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: robotics

### Summary
Do not transition OpenArmX grasp stages before the visible TCP light column is centered on the cube.

### Details
The user observed that the green light column was still not centered and asked to align the column first, then grasp. The earlier timeout-based state transitions could move from alignment to approach/descent before the TCP XY error was small enough.

### Suggested Action
Gate stage transitions on explicit TCP XY error to the cube center; avoid timeout-based forced transitions for alignment-sensitive stages.

### Metadata
- Source: user_feedback
- Related Files: /home/y/Desktop/openarmx_robosuite_example/run_openarmx_standalone_viewer.py
- Tags: robosuite, openarmx, grasping, tcp-alignment

---

## [LRN-20260512-001] correction

**Logged**: 2026-05-12T18:44:16+08:00
**Priority**: high
**Status**: pending
**Area**: robotics

### Summary
The "good" OpenArmX robosuite viewer version keeps `--no-assisted-lift` as a legacy visual-grasp command.

### Details
The user confirmed this version is good after fixing two issues: the gripper close target must map desired total inner jaw gap to a single-finger qpos, and the old `--no-assisted-lift` command should not switch the demo back to raw pure-physics behavior. Strict unassisted contact debugging belongs behind `--pure-physics`.

### Suggested Action
Do not revert `--no-assisted-lift` to pure physics. Preserve `demo_cube_gripper_close_qpos()` returning half the desired inner gap per finger, keep the visual centered-grasp path as the default / legacy command behavior, and use `--pure-physics` only for raw contact experiments.

### Metadata
- Source: user_feedback
- Related Files: /home/y/Desktop/openarmx_robosuite_example/run_openarmx_standalone_viewer.py
- Tags: robosuite, openarmx, grasping, visual-grasp, pure-physics

---

## [LRN-20260509-001] correction

**Logged**: 2026-05-09T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: robotics

### Summary
Low-stage XY locking made the OpenArmX cube grasp worse.

### Details
The user reported the previous version was closer to success, but the later change that locked XY during descent and reduced low-stage XY/orientation weights caused the gripper to miss the cube entirely. Do not reuse that strategy without instrumentation because it can lock in a pre-grasp alignment error.

### Suggested Action
Prefer the earlier continuous target tracking version, and tune speed / grasp height in smaller single-variable changes.

### Metadata
- Source: user_feedback
- Related Files: /home/y/Desktop/openarmx_robosuite_example/run_openarmx_standalone_viewer.py
- Tags: robosuite, openarmx, grasping, ik

---
