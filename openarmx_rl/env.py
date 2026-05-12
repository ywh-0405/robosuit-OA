from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

import run_openarmx_standalone_viewer as viewer_demo


STATE_NAMES = ("center_xy", "approach", "descend", "close", "settle", "lift")


class OpenArmXVisualGraspEnv:
    action_dim = 4
    observation_dim = 29

    def __init__(
        self,
        max_steps=220,
        action_pos_scale=0.03,
        action_gripper_scale=0.02,
        success_lift_height=0.12,
        success_center_tolerance=0.02,
        assisted_lift=True,
    ):
        self.max_steps = int(max_steps)
        self.action_pos_scale = float(action_pos_scale)
        self.action_gripper_scale = float(action_gripper_scale)
        self.success_lift_height = float(success_lift_height)
        self.success_center_tolerance = float(success_center_tolerance)
        self.assisted_lift = bool(assisted_lift)
        self.sim = None
        self.robot_model = None
        self.cube_joint_name = None
        self.arm_joint_names = None
        self.gripper_joint_names = None
        self.gripper_actuator_ids = None
        self.demo = None
        self.step_count = 0
        self.initial_cube_z = 0.0
        self.contact_hold_pos = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.sim, self.robot_model = viewer_demo.build_standalone_sim()
        self.cube_joint_name = viewer_demo.find_free_joint_name(self.sim, "cube")
        self.arm_joint_names = viewer_demo.find_arm_joint_names(self.robot_model, "right")
        self.gripper_joint_names = viewer_demo.find_gripper_joint_names(self.robot_model, "right")
        self.gripper_actuator_ids = viewer_demo.find_gripper_actuator_ids(self.sim, "right")
        viewer_demo.set_gripper_qpos(
            self.sim,
            viewer_demo.find_gripper_joint_names(self.robot_model, "left"),
            viewer_demo.OPEN_GRIPPER_QPOS,
        )
        viewer_demo.set_gripper_ctrl(
            self.sim,
            viewer_demo.find_gripper_actuator_ids(self.sim, "left"),
            viewer_demo.OPEN_GRIPPER_QPOS,
        )
        self.demo = viewer_demo.IKGraspDemo(
            self.sim,
            self.robot_model,
            "right",
            self.cube_joint_name,
        )
        self.step_count = 0
        self.contact_hold_pos = None
        self.initial_cube_z = float(viewer_demo.free_joint_pos(self.sim, self.cube_joint_name)[2])
        obs = self._observation()
        return obs, self._info(False, 0.0)

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        target_pos, target_xmat, expert_gripper_qpos = self.demo.target_and_gripper()
        target_pos = np.asarray(target_pos, dtype=float) + action[:3] * self.action_pos_scale
        gripper_qpos = float(
            np.clip(
                expert_gripper_qpos + action[3] * self.action_gripper_scale,
                viewer_demo.CLOSED_GRIPPER_QPOS,
                viewer_demo.OPEN_GRIPPER_QPOS,
            )
        )

        position_error = 0.0
        rotation_error = 0.0
        for _ in range(viewer_demo.DEFAULT_IK_ITERS):
            position_error, rotation_error = viewer_demo.ik_step_to_target(
                self.sim,
                self.robot_model,
                "right",
                target_pos,
                target_xmat=target_xmat,
                damping=viewer_demo.DEFAULT_IK_DAMPING,
                gain=viewer_demo.DEFAULT_IK_GAIN,
                max_step=viewer_demo.DEFAULT_IK_MAX_STEP,
                orientation_weight=viewer_demo.DEFAULT_ORIENTATION_WEIGHT,
            )

        viewer_demo.set_gripper_qpos(self.sim, self.gripper_joint_names, gripper_qpos)
        viewer_demo.set_gripper_ctrl(self.sim, self.gripper_actuator_ids, gripper_qpos)
        self.contact_hold_pos = viewer_demo.apply_visual_grasp_mode(
            self.sim,
            self.robot_model,
            "right",
            self.cube_joint_name,
            self.demo.state,
            self.assisted_lift,
            self.contact_hold_pos,
        )
        control_point = viewer_demo.gripper_control_point(self.sim, self.robot_model, "right")
        self.demo.update_state(position_error, rotation_error, control_point)

        self.sim.step()
        self.step_count += 1

        obs = self._observation()
        success = self._success()
        reward = self._reward(success)
        done = bool(success)
        truncated = bool(self.step_count >= self.max_steps and not done)
        return obs, reward, done, truncated, self._info(success, reward)

    def close(self):
        self.sim = None
        self.robot_model = None
        self.demo = None

    def _observation(self):
        arm_qpos = viewer_demo.get_joint_qpos_vector(self.sim, self.arm_joint_names)
        gripper_qpos = viewer_demo.get_joint_qpos_vector(self.sim, self.gripper_joint_names)
        cube_pos = viewer_demo.free_joint_pos(self.sim, self.cube_joint_name)
        pad_center = viewer_demo.gripper_fingerpad_center(self.sim, self.robot_model, "right")
        target_pos, _, gripper_target = self.demo.target_and_gripper()
        state_one_hot = np.zeros(len(STATE_NAMES), dtype=np.float32)
        state_one_hot[STATE_NAMES.index(self.demo.state)] = 1.0
        obs = np.concatenate(
            [
                arm_qpos,
                gripper_qpos,
                cube_pos,
                pad_center,
                cube_pos - pad_center,
                np.asarray(target_pos, dtype=float),
                np.array([gripper_target], dtype=float),
                np.array([self.step_count / max(1, self.max_steps)], dtype=float),
                state_one_hot,
            ]
        ).astype(np.float32)
        if obs.shape != (self.observation_dim,):
            raise RuntimeError(f"Observation shape drifted to {obs.shape}")
        return obs

    def _success(self):
        cube_pos = viewer_demo.free_joint_pos(self.sim, self.cube_joint_name)
        pad_center = viewer_demo.gripper_fingerpad_center(self.sim, self.robot_model, "right")
        lifted = cube_pos[2] >= self.initial_cube_z + self.success_lift_height
        centered = np.linalg.norm(cube_pos - pad_center) <= self.success_center_tolerance
        return bool(lifted and centered)

    def _reward(self, success):
        cube_pos = viewer_demo.free_joint_pos(self.sim, self.cube_joint_name)
        pad_center = viewer_demo.gripper_fingerpad_center(self.sim, self.robot_model, "right")
        center_dist = float(np.linalg.norm(cube_pos - pad_center))
        lift = float(max(0.0, cube_pos[2] - self.initial_cube_z))
        return float(3.0 * lift - center_dist + (5.0 if success else 0.0))

    def _info(self, success, reward):
        return {
            "success": bool(success),
            "state": self.demo.state if self.demo is not None else None,
            "step": int(self.step_count),
            "reward": float(reward),
            "cube_pos": viewer_demo.free_joint_pos(self.sim, self.cube_joint_name).astype(np.float32),
            "fingerpad_center": viewer_demo.gripper_fingerpad_center(
                self.sim, self.robot_model, "right"
            ).astype(np.float32),
        }


def expert_action(_obs=None, _info=None):
    return np.zeros(OpenArmXVisualGraspEnv.action_dim, dtype=np.float32)


def save_episode_npz(output_dir, episode_index, episode, prefix="openarmx_visual_grasp"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    path = output_dir / f"{prefix}_ep{episode_index:03d}_{timestamp}.npz"
    arrays = {
        "obs": np.asarray(episode["obs"], dtype=np.float32),
        "actions": np.asarray(episode["actions"], dtype=np.float32),
        "rewards": np.asarray(episode["rewards"], dtype=np.float32),
        "next_obs": np.asarray(episode["next_obs"], dtype=np.float32),
        "dones": np.asarray(episode["dones"], dtype=np.bool_),
        "success": np.asarray([bool(episode.get("success", False))], dtype=np.bool_),
        "episode_length": np.asarray([int(episode.get("episode_length", len(episode["obs"])))], dtype=np.int32),
    }
    np.savez_compressed(path, **arrays)
    return path
