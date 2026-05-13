import argparse
from pathlib import Path

import numpy as np

from openarmx_rl.ddpg_bc import DDPGAgent, DDPGConfig, require_torch, set_seed
from openarmx_rl.env import OpenArmXVisualGraspEnv


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train OpenArmX visual-grasp policy with DDPG+BC.")
    parser.add_argument("--dataset-dir", default="openarmx_visual_grasp_dataset")
    parser.add_argument("--checkpoint", default="checkpoints/openarmx_visual_actor.pth")
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--random-cube-pos", action="store_true", default=False)
    parser.add_argument("--cube-x-range", type=float, nargs=2, default=[-0.305, -0.270])
    parser.add_argument("--cube-y-range", type=float, nargs=2, default=[-0.16, -0.12])
    return parser


def bc_weight_for_step(step, total_steps, start=1.0, end=0.2):
    if total_steps <= 0:
        return end
    ratio = min(1.0, max(0.0, step / float(total_steps)))
    return float(start + (end - start) * ratio)


def train(args):
    torch = require_torch()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = OpenArmXVisualGraspEnv(
        max_steps=args.max_episode_steps,
        randomize_cube_pos=args.random_cube_pos,
        cube_x_range=args.cube_x_range,
        cube_y_range=args.cube_y_range,
    )
    cfg = DDPGConfig(
        seed=args.seed,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        pretrain_steps=args.pretrain_steps,
        total_steps=args.total_steps,
        exploration_noise=args.exploration_noise,
        reward_scale=args.reward_scale,
    )
    agent = DDPGAgent(env.observation_dim, env.action_dim, cfg, device)

    loaded = agent.rb.load_dataset(args.dataset_dir, reward_scale=args.reward_scale)
    print(f"loaded_transitions={loaded} dataset_dir={args.dataset_dir}")
    if loaded <= 0:
        raise SystemExit(
            f"No demonstration transitions found in {args.dataset_dir}. "
            "Run collect_openarmx_data.py first."
        )

    for step in range(args.pretrain_steps):
        info = agent.update(use_bc_loss=True, bc_weight=1.0)
        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print(f"pretrain_step={step + 1} info={info}")

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    for step in range(args.total_steps):
        bc_weight = bc_weight_for_step(step, args.total_steps)
        action = agent.act(obs, noise=args.exploration_noise)
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = bool(done or truncated)
        agent.rb.add(obs, action, reward * args.reward_scale, next_obs, terminal)
        update_info = agent.update(use_bc_loss=True, bc_weight=bc_weight)
        episode_reward += float(reward)
        obs = next_obs

        if terminal:
            print(
                f"episode_end step={step + 1} reward={episode_reward:.3f} "
                f"success={info['success']} bc_weight={bc_weight:.3f}"
            )
            obs, _ = env.reset()
            episode_reward = 0.0

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print(f"train_step={step + 1} info={update_info}")

    env.close()
    checkpoint = Path(args.checkpoint)
    agent.save_actor(checkpoint)
    print(f"saved_actor={checkpoint}")
    return checkpoint


def main():
    args = build_arg_parser().parse_args()
    try:
        train(args)
    except RuntimeError as exc:
        if "PyTorch is required" in str(exc):
            raise SystemExit(str(exc))
        raise


if __name__ == "__main__":
    main()
