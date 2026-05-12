import argparse

import numpy as np

from openarmx_rl.env import OpenArmXVisualGraspEnv, expert_action, save_episode_npz


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect OpenArmX visual-grasp expert episodes.")
    parser.add_argument("--output-dir", default="openarmx_visual_grasp_dataset")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--render", action="store_true", default=False)
    return parser


def collect_episode(env, noise=0.0):
    obs, info = env.reset()
    episode = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
    done = False
    truncated = False
    success = False

    while not done and not truncated:
        action = expert_action(obs, info)
        if noise > 0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = bool(done or truncated)
        episode["obs"].append(obs)
        episode["actions"].append(action)
        episode["rewards"].append(reward)
        episode["next_obs"].append(next_obs)
        episode["dones"].append(terminal)
        success = success or bool(info["success"])
        obs = next_obs

    episode["success"] = success
    episode["episode_length"] = len(episode["obs"])
    return episode


def main():
    args = build_arg_parser().parse_args()
    env = OpenArmXVisualGraspEnv(max_steps=args.max_steps)
    for episode_index in range(args.episodes):
        episode = collect_episode(env, noise=args.noise)
        path = save_episode_npz(args.output_dir, episode_index, episode)
        print(
            f"episode {episode_index + 1}/{args.episodes} "
            f"steps={episode['episode_length']} success={episode['success']} saved={path}"
        )
    env.close()


if __name__ == "__main__":
    main()
