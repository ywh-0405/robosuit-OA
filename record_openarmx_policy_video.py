import argparse
from pathlib import Path

import numpy as np

import run_openarmx_standalone_viewer as viewer_demo
from evaluate_openarmx_policy import load_actor, policy_action
from openarmx_rl.ddpg_bc import require_torch, set_seed
from openarmx_rl.env import OpenArmXVisualGraspEnv
from robosuite.utils.binding_utils import MjRenderContextOffscreen


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Record an OpenArmX policy rollout to video.")
    parser.add_argument("--checkpoint", default="checkpoints/openarmx_visual_actor.pth")
    parser.add_argument("--output", default="openarmx_policy_rollout.mp4")
    parser.add_argument("--frames-dir", default="openarmx_policy_frames")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def configure_offscreen_camera(render_context):
    render_context.cam.type = viewer_demo.mujoco.mjtCamera.mjCAMERA_FREE
    render_context.cam.lookat[:] = [-0.18, -0.10, 0.92]
    render_context.cam.distance = 1.05
    render_context.cam.azimuth = 158
    render_context.cam.elevation = -32
    for flag in (
        viewer_demo.mujoco.mjtVisFlag.mjVIS_ACTUATOR,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_JOINT,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_RANGEFINDER,
        viewer_demo.mujoco.mjtVisFlag.mjVIS_TENDON,
    ):
        render_context.vopt.flags[flag] = 0


def render_frame(render_context, width, height):
    render_context.render(width=width, height=height)
    return render_context.read_pixels(width=width, height=height)[::-1]


def hide_debug_sites(sim):
    for site_id, site_name in enumerate(sim.model.site_names):
        site_size = sim.model.site_size[site_id]
        site_rgba = sim.model.site_rgba[site_id]
        if "grip_site_cylinder" in site_name or max(site_size) > 0.05 or site_rgba[3] < 0.05:
            sim.model.site_rgba[site_id, 3] = 0.0


def save_png(path, frame):
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def record_episode(args, actor, device, writer, episode_index, frames_dir):
    env = OpenArmXVisualGraspEnv(max_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed + episode_index)
    hide_debug_sites(env.sim)

    render_context = MjRenderContextOffscreen(
        env.sim,
        device_id=-1,
        max_width=args.width,
        max_height=args.height,
    )
    configure_offscreen_camera(render_context)

    total_reward = 0.0
    success = False
    mid_saved = False
    steps = 0
    episode_dir = frames_dir / f"episode_{episode_index + 1:02d}"

    try:
        done = False
        truncated = False
        while not done and not truncated:
            frame = render_frame(render_context, args.width, args.height)
            if steps == 0:
                save_png(episode_dir / "start.png", frame)
            if not mid_saved and steps >= args.max_steps // 2:
                save_png(episode_dir / "middle.png", frame)
                mid_saved = True
            writer.write(frame[:, :, ::-1])

            action = policy_action(actor, obs, device)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info["success"])
            steps += 1

        final_frame = render_frame(render_context, args.width, args.height)
        writer.write(final_frame[:, :, ::-1])
        save_png(episode_dir / "final.png", final_frame)
    finally:
        env.close()

    return {
        "episode": episode_index + 1,
        "steps": steps,
        "reward": total_reward,
        "success": success,
    }


def record(args):
    cv2 = __import__("cv2")
    torch = require_torch()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    probe_env = OpenArmXVisualGraspEnv(max_steps=args.max_steps)
    actor = load_actor(args.checkpoint, probe_env.observation_dim, probe_env.action_dim, device)
    probe_env.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    results = []
    try:
        for episode_index in range(args.episodes):
            result = record_episode(args, actor, device, writer, episode_index, frames_dir)
            results.append(result)
            print(
                f"episode={result['episode']}/{args.episodes} "
                f"steps={result['steps']} reward={result['reward']:.3f} "
                f"success={result['success']}"
            )
    finally:
        writer.release()

    successes = sum(1 for result in results if result["success"])
    mean_reward = float(np.mean([result["reward"] for result in results])) if results else 0.0
    mean_steps = float(np.mean([result["steps"] for result in results])) if results else 0.0
    print(
        f"saved_video={output_path} frames_dir={frames_dir} "
        f"episodes={args.episodes} successes={successes}/{args.episodes} "
        f"mean_reward={mean_reward:.3f} mean_steps={mean_steps:.1f}"
    )
    return output_path


def main():
    args = build_arg_parser().parse_args()
    record(args)


if __name__ == "__main__":
    main()
