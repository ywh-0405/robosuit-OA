import argparse
import time

import numpy as np
import robosuite as suite


def build_env_kwargs():
    return {
        "env_name": "TwoArmLift",
        "robots": "OpenArmX",
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "reward_shaping": True,
        "ignore_done": True,
        "control_freq": 20,
        "horizon": 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenArmX in robosuite with an on-screen MuJoCo viewer.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of simulation steps to run.")
    parser.add_argument("--max-fr", type=float, default=25.0, help="Maximum viewer frame rate.")
    parser.add_argument("--camera-id", type=int, default=0, help="MuJoCo viewer camera id.")
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="Use random actions instead of zeros so the robot and object move.",
    )
    args = parser.parse_args()

    env = suite.make(**build_env_kwargs())

    try:
        obs = env.reset()
        print("env_type =", type(env).__name__)
        print("robots =", [robot.name for robot in env.robots])
        print("action_dim =", env.action_dim)
        print("obs_keys_sample =", sorted(list(obs.keys()))[:5])
        print("viewer_camera_id =", args.camera_id)
        print("random_action =", args.random_action)

        if getattr(env, "viewer", None) is not None:
            env.viewer.set_camera(camera_id=args.camera_id)

        low, high = env.action_spec

        for _ in range(args.steps):
            start = time.time()
            if args.random_action:
                action = np.random.uniform(low, high)
            else:
                action = np.zeros(env.action_dim)

            env.step(action)
            env.render()

            elapsed = time.time() - start
            delay = 1.0 / args.max_fr - elapsed
            if delay > 0:
                time.sleep(delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
