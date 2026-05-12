import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module_from_repo(filename, module_name):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_grasp_env_reset_and_step_shapes():
    from openarmx_rl.env import OpenArmXVisualGraspEnv, expert_action

    env = OpenArmXVisualGraspEnv(max_steps=20)
    obs, info = env.reset()

    assert obs.shape == (env.observation_dim,)
    assert obs.dtype == np.float32
    assert info["state"] == "center_xy"

    action = expert_action(obs, info)
    next_obs, reward, done, truncated, step_info = env.step(action)

    assert action.shape == (env.action_dim,)
    assert next_obs.shape == (env.observation_dim,)
    assert next_obs.dtype == np.float32
    assert isinstance(float(reward), float)
    assert done in (True, False)
    assert truncated in (True, False)
    assert "success" in step_info
    assert "cube_pos" in step_info
    assert "fingerpad_center" in step_info


def test_save_episode_npz_writes_ur5e_compatible_fields(tmp_path):
    from openarmx_rl.env import save_episode_npz

    episode = {
        "obs": [np.zeros(29, dtype=np.float32)],
        "actions": [np.zeros(4, dtype=np.float32)],
        "rewards": [1.0],
        "next_obs": [np.ones(29, dtype=np.float32)],
        "dones": [True],
        "success": True,
        "episode_length": 1,
    }

    path = save_episode_npz(tmp_path, 3, episode, prefix="openarmx_test")

    assert path.exists()
    data = np.load(path)
    assert set(data.files) >= {
        "obs",
        "actions",
        "rewards",
        "next_obs",
        "dones",
        "success",
        "episode_length",
    }
    assert data["obs"].shape == (1, 29)
    assert data["actions"].shape == (1, 4)
    assert bool(data["success"][0]) is True


def test_collect_parser_defaults_are_small_and_headless():
    module = load_module_from_repo("collect_openarmx_data.py", "collect_openarmx_data")

    args = module.build_arg_parser().parse_args([])

    assert args.episodes <= 10
    assert args.render is False
    assert args.output_dir == "openarmx_visual_grasp_dataset"


def test_collect_episode_noise_does_not_require_env_rng():
    module = load_module_from_repo("collect_openarmx_data.py", "collect_openarmx_data_noise")

    class OneStepEnv:
        def reset(self):
            return np.zeros(29, dtype=np.float32), {"state": "center_xy"}

        def step(self, action):
            assert action.shape == (4,)
            return (
                np.ones(29, dtype=np.float32),
                1.0,
                True,
                False,
                {"success": True},
            )

    episode = module.collect_episode(OneStepEnv(), noise=0.01)

    assert episode["success"] is True
    assert episode["episode_length"] == 1
    assert np.asarray(episode["actions"]).shape == (1, 4)


def test_train_parser_defaults_are_smoke_sized():
    module = load_module_from_repo("train_openarmx_ddpg_bc.py", "train_openarmx_ddpg_bc")

    args = module.build_arg_parser().parse_args([])

    assert args.dataset_dir == "openarmx_visual_grasp_dataset"
    assert args.checkpoint == "checkpoints/openarmx_visual_actor.pth"
    assert args.pretrain_steps <= 200
    assert args.total_steps <= 1000
    assert args.batch_size <= 128


def test_evaluate_parser_defaults_are_headless_and_short():
    module = load_module_from_repo("evaluate_openarmx_policy.py", "evaluate_openarmx_policy")

    args = module.build_arg_parser().parse_args([])

    assert args.checkpoint == "checkpoints/openarmx_visual_actor.pth"
    assert args.episodes <= 10
    assert args.render is False


def test_replay_buffer_loads_episode_npz(tmp_path):
    from openarmx_rl.ddpg_bc import ReplayBuffer
    from openarmx_rl.env import save_episode_npz

    episode = {
        "obs": [np.zeros(29, dtype=np.float32), np.ones(29, dtype=np.float32)],
        "actions": [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32) * 0.5],
        "rewards": [0.0, 1.0],
        "next_obs": [np.ones(29, dtype=np.float32), np.ones(29, dtype=np.float32) * 2],
        "dones": [False, True],
        "success": True,
        "episode_length": 2,
    }
    save_episode_npz(tmp_path, 0, episode)

    buffer = ReplayBuffer(buffer_size=10, state_dim=29, action_dim=4)
    loaded = buffer.load_dataset(tmp_path)

    assert loaded == 2
    assert buffer.size == 2
    batch = buffer.sample(2)
    assert batch[0].shape == (2, 29)
    assert batch[1].shape == (2, 4)


def test_actor_outputs_bounded_actions():
    torch = pytest.importorskip("torch")

    from openarmx_rl.ddpg_bc import Actor

    actor = Actor(state_dim=29, action_dim=4)
    obs = torch.zeros((3, 29), dtype=torch.float32)

    action = actor(obs)

    assert action.shape == (3, 4)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
