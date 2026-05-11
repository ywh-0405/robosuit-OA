import robosuite as suite


def main():
    env = suite.make(
        env_name="TwoArmLift",
        robots="OpenArmX",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        ignore_done=True,
        horizon=10,
    )

    try:
        obs = env.reset()
        print("env_type =", type(env).__name__)
        print("robots =", [robot.name for robot in env.robots])
        print("action_dim =", env.action_dim)
        print("obs_keys_sample =", sorted(list(obs.keys()))[:5])
    finally:
        env.close()


if __name__ == "__main__":
    main()
