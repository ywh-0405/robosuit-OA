import argparse
import time
from contextlib import nullcontext

import numpy as np

import run_openarmx_standalone_viewer as viewer_demo
from openarmx_rl.ddpg_bc import Actor, require_torch, set_seed
from openarmx_rl.env import OpenArmXVisualGraspEnv


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate an OpenArmX visual-grasp actor.")
    parser.add_argument("--checkpoint", default="checkpoints/openarmx_visual_actor.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each rendered policy step. Use with --render for slower playback.",
    )
    parser.add_argument("--random-cube-pos", action="store_true", default=False)
    parser.add_argument("--cube-x-range", type=float, nargs=2, default=[-0.305, -0.270])
    parser.add_argument("--cube-y-range", type=float, nargs=2, default=[-0.16, -0.12])
    return parser


def policy_action(actor, obs, device):
    torch = require_torch()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = actor(obs_t).cpu().numpy()[0]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def load_actor(checkpoint, state_dim, action_dim, device):
    torch = require_torch()
    actor = Actor(state_dim, action_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device))
    actor.eval()
    return actor


def maybe_viewer(env, enabled):
    if not enabled:
        return nullcontext(None)
    return viewer_demo.viewer.launch_passive(
        env.sim.model._model,
        env.sim.data._data,
        show_left_ui=False,
        show_right_ui=False,
    )


def evaluate(args):
    torch = require_torch()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = OpenArmXVisualGraspEnv(
        max_steps=args.max_steps,
        randomize_cube_pos=args.random_cube_pos,
        cube_x_range=args.cube_x_range,
        cube_y_range=args.cube_y_range,
    )
    actor = load_actor(args.checkpoint, env.observation_dim, env.action_dim, device)

    rewards = []
    lengths = []
    successes = 0
    for episode_index in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_index)
        total_reward = 0.0
        success = False
        steps = 0
        with maybe_viewer(env, args.render) as vwr:
            if vwr is not None:
                viewer_demo.configure_camera(vwr, "free")
            done = False
            truncated = False
            while not done and not truncated:
                action = policy_action(actor, obs, device)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                success = success or bool(info["success"])
                steps += 1
                if vwr is not None:
                    vwr.sync()
                    if args.render_delay > 0:
                        time.sleep(args.render_delay)

        rewards.append(total_reward)
        lengths.append(steps)
        successes += int(success)
        print(
            f"episode={episode_index + 1}/{args.episodes} "
            f"reward={total_reward:.3f} steps={steps} success={success}"
        )

    env.close()
    rewards = np.asarray(rewards, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    success_rate = successes / max(1, args.episodes)
    print("========== Evaluation Summary ==========")
    print(f"episodes={args.episodes}")
    print(f"mean_reward={rewards.mean():.3f}")
    print(f"mean_episode_length={lengths.mean():.1f}")
    print(f"success_rate={success_rate:.3f}")
    return {
        "mean_reward": float(rewards.mean()),
        "mean_episode_length": float(lengths.mean()),
        "success_rate": float(success_rate),
    }


def main():
    args = build_arg_parser().parse_args()
    try:
        evaluate(args)
    except RuntimeError as exc:
        if "PyTorch is required" in str(exc):
            raise SystemExit(str(exc))
        raise


if __name__ == "__main__":
    main()
