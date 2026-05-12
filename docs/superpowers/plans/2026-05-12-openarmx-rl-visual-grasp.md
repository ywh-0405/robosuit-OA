# OpenArmX RL Visual Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal OpenArmX visual-grasp data collection, DDPG+BC training, and evaluation pipeline.

**Architecture:** Reuse the standalone OpenArmX MuJoCo scene and wrap it in a lightweight vector-observation environment. Keep data collection, training primitives, and CLI entry points separate so each part can be tested independently.

**Tech Stack:** Python, NumPy, MuJoCo/robosuite `MjSim`, PyTorch, pytest, compressed `.npz` episode files.

---

## File Structure

- Create `openarmx_rl/__init__.py`: package exports.
- Create `openarmx_rl/env.py`: `OpenArmXVisualGraspEnv`, expert-action helper, and episode writer.
- Create `openarmx_rl/ddpg_bc.py`: replay buffer, actor, critic, and agent update logic.
- Create `collect_openarmx_data.py`: CLI data collection.
- Create `train_openarmx_ddpg_bc.py`: CLI training.
- Create `evaluate_openarmx_policy.py`: CLI evaluation.
- Create `test_openarmx_rl_pipeline.py`: fast tests for env, dataset, replay, and actor.
- Modify `.gitignore`: ignore generated datasets/checkpoints/models.
- Modify `README.md`: document collection, training, and evaluation.

## Tasks

### Task 1: Environment And Dataset Format

**Files:**
- Create: `openarmx_rl/__init__.py`
- Create: `openarmx_rl/env.py`
- Test: `test_openarmx_rl_pipeline.py`

- [ ] Write failing tests for reset/step shapes and `.npz` episode fields.
- [ ] Implement `OpenArmXVisualGraspEnv`.
- [ ] Implement `expert_action()` returning a 4-D zero action.
- [ ] Implement `save_episode_npz()`.
- [ ] Run targeted tests.

### Task 2: DDPG+BC Core

**Files:**
- Create: `openarmx_rl/ddpg_bc.py`
- Test: `test_openarmx_rl_pipeline.py`

- [ ] Write failing tests for replay loading and actor action bounds.
- [ ] Implement `ReplayBuffer`.
- [ ] Implement `Actor`, `Critic`, and `DDPGAgent`.
- [ ] Run targeted tests.

### Task 3: CLI Scripts

**Files:**
- Create: `collect_openarmx_data.py`
- Create: `train_openarmx_ddpg_bc.py`
- Create: `evaluate_openarmx_policy.py`
- Test: `test_openarmx_rl_pipeline.py`

- [ ] Write failing tests for parser defaults.
- [ ] Implement data collection CLI.
- [ ] Implement training CLI.
- [ ] Implement evaluation CLI.
- [ ] Run targeted tests.

### Task 4: Docs And Verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] Add generated data/model ignores.
- [ ] Document collect/train/evaluate commands.
- [ ] Run full test suite.
- [ ] Run a one-episode data collection smoke test.
- [ ] Run a tiny training smoke test if PyTorch is available.
- [ ] Commit the completed pipeline.
